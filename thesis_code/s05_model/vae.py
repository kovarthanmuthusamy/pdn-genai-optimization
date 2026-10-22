"""The multi-modal VAE: three Gaussian experts fused by Product-of-Experts.

    heatmap (B,1,64,64) --CNN--+
    occupancy (B,52,T) --GNN---+--> PoE --> z (B,128) --> heatmap decoder (FiLM on freq, fused
    impedance (B,1,231) --GNN--+                         with a spatial occupancy tower)
                                                     --> occupancy decoder (GNN, (B,52,T) logits)
                                                     --> impedance decoder (GNN, (B,1,231))

Conditioning: K (number of decaps, 0..k_max) through an embedding, the PI frequency through
Fourier features. Every expert writes all 128 latent dims.

Usage (import only):
MultiInputVAE(latent_dim, cond_dim, freq_fourier_features, n_occ_classes, k_max, modality_dropout)
  .encode(hm, occ, imp, K, pi)          -> z, mu, logvar, expert_stats   (training / full encode)
  .decode(z, K, pi, occupancy=None)     -> heatmap, occupancy logits, impedance
  .forward(hm, occ, imp, K, pi)         -> decode(encode(...)) outputs + mu, logvar, expert_stats
  ._layout_latent_slice(z)              the part of z the impedance decoder reads (all of it)
  .heatmap_private_dim / .shared_latent_dim / .layout_latent_dim   latent sizes read by the search

Training switches set by the loop: ``modality_dropout`` and ``modality_dropout_protect_heatmap``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from thesis_code.common.placement import (LABEL_TO_COORD, LABELS_ORDERED, hard_onehot_from_probs, hard_presence,
                                          presence_from_onehot)
from thesis_code.s05_model.blocks import FiLM2d, FreqConditioner, OccSpatialTower, ResidualConv2d, SelfAttn2d, UpsampleConv2d
from thesis_code.s05_model.graphs import ImpSpectrumGraphDecoder, ImpSpectrumGraphEncoder, OccGraphDecoder, OccGraphEncoder

HM_BOTTLENECK = 8     # heatmap decoder starts from an 8x8 map
HM_DROPOUT = 0.05     # light dropout on the heatmap conv path (keeps peaks and gradients)
EXPERT_LOGVAR_MIN, EXPERT_LOGVAR_MAX = -6.0, 1.0   # caps each expert's sigma at ~1.65
POSTERIOR_LOGVAR_MIN, POSTERIOR_LOGVAR_MAX = -4.0, 2.0
EXPERTS = ("heatmap", "occupancy", "impedance")
OCC_SPATIAL_CH = 8    # channels of the spatial occupancy tower at each decoder scale
GRAPH_HIDDEN, GRAPH_LAYERS, GRAPH_DROPOUT = 128, 3, 0.1   # occupancy / impedance graph nets


class MultiInputVAE(nn.Module):
    """Heatmap / occupancy / impedance VAE conditioned on K and PI frequency (see module doc)."""

    def __init__(
        self,
        latent_dim: int = 128,
        cond_dim: int = 32,
        freq_fourier_features: int = 48,
        n_occ_classes: int = 3,
        k_max: int = 10,
        modality_dropout: float = 0.0,
    ):
        super().__init__()
        if n_occ_classes < 1:
            raise ValueError(f"n_occ_classes must be >= 1 (number of decap types), got {n_occ_classes}")
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.n_occ_classes = int(n_occ_classes)
        self.k_max = int(k_max)
        occ_spatial_ch = OCC_SPATIAL_CH
        graph_hidden_dim, graph_num_layers, graph_dropout = GRAPH_HIDDEN, GRAPH_LAYERS, GRAPH_DROPOUT
        # No private/shared split: every expert and decoder sees the whole latent.
        self.heatmap_private_dim = 0
        self.shared_latent_dim = latent_dim
        self.layout_latent_dim = latent_dim
        # Training-time switches (set by the training loop).
        self.modality_dropout = modality_dropout
        self.modality_dropout_protect_heatmap = False

        occ_hw = torch.tensor([LABEL_TO_COORD[label] for label in LABELS_ORDERED], dtype=torch.long)
        self.register_buffer("_occ_label_hw", occ_hw, persistent=False)

        # Construction order below is the parameter-initialisation order; keep it as is so a
        # fresh run with the same seed starts from the same weights.
        self.k_embedding = nn.Embedding(self.k_max + 1, cond_dim)   # K in 0..k_max
        self.freq_conditioner = FreqConditioner(cond_dim, num_fourier=freq_fourier_features)

        # ── Encoders ─────────────────────────────────────────────────────────
        self.heatmap_enc1a = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
        )
        self.heatmap_enc1b = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(),
            nn.Dropout2d(HM_DROPOUT),
        )
        self.heatmap_enc_attn = SelfAttn2d(32, num_heads=1)
        self.heatmap_enc_film = FiLM2d(32, 2 * cond_dim)
        self.heatmap_enc2_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        self.heatmap_enc2_res = ResidualConv2d(128)
        self.heatmap_enc2_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * HM_BOTTLENECK * HM_BOTTLENECK, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Dropout(HM_DROPOUT),
            nn.Linear(512, 64),
        )
        self.occupancy_encoder = OccGraphEncoder(
            out_dim=64, hidden_dim=graph_hidden_dim, num_layers=graph_num_layers,
            dropout=graph_dropout, n_occ_classes=self.n_occ_classes,
        )
        self.impedance_encoder = ImpSpectrumGraphEncoder(
            out_dim=64, hidden_dim=graph_hidden_dim, num_layers=graph_num_layers, dropout=graph_dropout,
        )
        self.layout_joint_fusion = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
        )

        # ── Expert heads: heatmap sees K + frequency, occupancy/impedance see K only ──
        enc_in_hm = 64 + 2 * cond_dim
        enc_in_k = 64 + cond_dim
        self.heatmap_mu = nn.Linear(enc_in_hm, latent_dim)
        self.heatmap_logvar = nn.Linear(enc_in_hm, latent_dim)
        self.occupancy_mu = nn.Linear(enc_in_k, latent_dim)
        self.occupancy_logvar = nn.Linear(enc_in_k, latent_dim)
        self.impedance_mu = nn.Linear(enc_in_k, latent_dim)
        self.impedance_logvar = nn.Linear(enc_in_k, latent_dim)

        # ── Heatmap decoder: z + cond -> 8x8 -> 16 -> 32 -> 64, FiLM at every scale ──
        self.heatmap_fc = nn.Sequential(
            nn.Linear(latent_dim + 2 * cond_dim, 512),
            nn.LeakyReLU(),
            nn.Dropout(HM_DROPOUT),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Dropout(HM_DROPOUT),
            nn.Linear(512, 128 * HM_BOTTLENECK * HM_BOTTLENECK),
            nn.LayerNorm(128 * HM_BOTTLENECK * HM_BOTTLENECK),
            nn.LeakyReLU(),
        )
        self.heatmap_dec_refine8 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(),
        )
        self.heatmap_dec_up1 = UpsampleConv2d(128, 64, scale=2, dropout=HM_DROPOUT)
        self.heatmap_dec_refine16 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(),
        )
        self.heatmap_dec_up2 = UpsampleConv2d(64, 32, scale=2, dropout=HM_DROPOUT)
        self.heatmap_dec_refine32 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(),
        )
        self.occ_spatial_tower = OccSpatialTower(occ_ch=occ_spatial_ch)
        self.heatmap_dec_occ_fuse16 = nn.Sequential(
            nn.Conv2d(64 + occ_spatial_ch, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(),
        )
        self.heatmap_dec_occ_fuse32 = nn.Sequential(
            nn.Conv2d(32 + occ_spatial_ch, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(),
        )
        self.heatmap_dec_occ_fuse64 = nn.Sequential(
            nn.Conv2d(16 + occ_spatial_ch, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.LeakyReLU(),
        )
        self.heatmap_dec_res = ResidualConv2d(32)
        self.heatmap_dec_attn = SelfAttn2d(32, num_heads=1)
        self.heatmap_dec_film = FiLM2d(32, 2 * cond_dim)
        self.heatmap_dec_film_bottleneck = FiLM2d(128, 2 * cond_dim)   # 8x8
        self.heatmap_dec_film16 = FiLM2d(64, 2 * cond_dim)             # 16x16
        self.heatmap_dec_film64 = FiLM2d(16, 2 * cond_dim)             # 64x64
        self.heatmap_dec_up3 = UpsampleConv2d(32, 16, scale=2, dropout=HM_DROPOUT)
        self.heatmap_dec_refine64 = ResidualConv2d(16)
        self.heatmap_dec_out = nn.Conv2d(16, 1, kernel_size=3, padding=1)

        # ── Occupancy / impedance decoders ──────────────────────────────────
        self.occupancy_decoder = OccGraphDecoder(
            dec_in=latent_dim + cond_dim, hidden_dim=graph_hidden_dim, num_layers=graph_num_layers,
            dropout=graph_dropout, n_occ_classes=self.n_occ_classes,
        )
        # z + K + a 64-d graph encoding of the (given or decoded) occupancy
        self.impedance_decoder = ImpSpectrumGraphDecoder(
            dec_in=latent_dim + cond_dim + 64, hidden_dim=graph_hidden_dim,
            num_layers=graph_num_layers, dropout=graph_dropout,
        )

    # ── Building blocks ───────────────────────────────────────────────────────

    def product_of_experts(self, mu_list, logvar_list, eps=1e-8):
        """Precision-weighted fusion of Gaussian experts plus a unit-Gaussian prior expert."""
        mu_list = mu_list + [torch.zeros_like(mu_list[0])]
        logvar_list = logvar_list + [torch.zeros_like(logvar_list[0])]
        precision_list = [1.0 / (torch.exp(lv) + eps) for lv in logvar_list]
        combined_var = 1.0 / (torch.stack(precision_list).sum(dim=0) + eps)
        combined_mu = combined_var * torch.stack(
            [mu * p for mu, p in zip(mu_list, precision_list)]
        ).sum(dim=0)
        return combined_mu, torch.log(combined_var + eps)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * torch.clamp(logvar, min=POSTERIOR_LOGVAR_MIN, max=POSTERIOR_LOGVAR_MAX))
        return mu + torch.randn_like(std) * std

    def _cond_emb(self, K: torch.Tensor, PI_freq: torch.Tensor) -> torch.Tensor:
        """(B,) K and (B,) normalised frequency -> (B, 2*cond_dim) = [K embedding | freq features]."""
        return torch.cat([self.k_embedding(K), self.freq_conditioner(PI_freq)], dim=1)

    def _encode_heatmap_feat(self, heatmap: torch.Tensor, cond_hm: torch.Tensor) -> torch.Tensor:
        h16 = self.heatmap_enc1b(self.heatmap_enc1a(heatmap))
        h16 = self.heatmap_enc_film(self.heatmap_enc_attn(h16), cond_hm)
        h8 = self.heatmap_enc2_res(self.heatmap_enc2_conv(h16))
        return self.heatmap_enc2_head(h8)

    def _layout_latent_slice(self, z: torch.Tensor) -> torch.Tensor:
        """The part of z the impedance decoder reads -- all of it here (no private split).
        Used by the latent search."""
        return z[:, : self.layout_latent_dim]

    def _encode_layout_joint(self, occupancy: torch.Tensor, impedance: torch.Tensor):
        """-> (occupancy feature, impedance feature, fused layout feature), each (B, 64)."""
        occ_feat = self.occupancy_encoder(occupancy)
        imp_feat = self.impedance_encoder(impedance)
        joint = self.layout_joint_fusion(torch.cat([occ_feat, imp_feat], dim=-1))
        return occ_feat, imp_feat, joint

    def _occupancy_to_grid7x8(self, occupancy: torch.Tensor) -> torch.Tensor:
        """(B,52) presence or (B,52,T) one-hot -> (B,1,7,8) presence grid on the board layout."""
        presence = presence_from_onehot(occupancy) if occupancy.dim() == 3 else occupancy
        grid = torch.zeros(presence.shape[0], 1, 7, 8, device=presence.device, dtype=presence.dtype)
        hw = self._occ_label_hw
        for i in range(hw.shape[0]):
            grid[:, 0, int(hw[i, 0]), int(hw[i, 1])] = presence[:, i]
        return grid

    def _decode_heatmap_tensor(self, z: torch.Tensor, cond: torch.Tensor,
                               occupancy: torch.Tensor | None = None) -> torch.Tensor:
        """z + cond -> (B,1,64,64). With ``occupancy`` the layout is fused in at 16/32/64 px."""
        h = self.heatmap_fc(torch.cat([z, cond], dim=1)).view(-1, 128, HM_BOTTLENECK, HM_BOTTLENECK)
        occ_feats = None if occupancy is None else self.occ_spatial_tower(self._occupancy_to_grid7x8(occupancy))

        h = self.heatmap_dec_film_bottleneck(self.heatmap_dec_refine8(h), cond)
        h = self.heatmap_dec_refine16(self.heatmap_dec_up1(h))
        if occ_feats is not None:
            h = self.heatmap_dec_occ_fuse16(torch.cat([h, occ_feats["16"]], dim=1))
        h = self.heatmap_dec_film16(h, cond)

        h = self.heatmap_dec_refine32(self.heatmap_dec_up2(h))
        if occ_feats is not None:
            h = self.heatmap_dec_occ_fuse32(torch.cat([h, occ_feats["32"]], dim=1))
        h = self.heatmap_dec_film(self.heatmap_dec_attn(self.heatmap_dec_res(h)), cond)

        h = self.heatmap_dec_up3(h)
        if occ_feats is not None:
            h = self.heatmap_dec_occ_fuse64(torch.cat([h, occ_feats["64"]], dim=1))
        h = self.heatmap_dec_film64(self.heatmap_dec_refine64(h), cond)
        return self.heatmap_dec_out(h)

    # ── Encode ───────────────────────────────────────────────────────────────

    def encode(self, heatmap, occupancy, impedance, K, PI_freq):
        """All three modalities -> (z, mu, logvar, expert_stats).

        heatmap (B,1,64,64); occupancy (B,52,T) one-hot, or legacy (B,52) presence read as
        type 1; impedance (B,1,231); K (B,) long; PI_freq (B,) normalised log-frequency.

        ``expert_stats`` holds each expert's (mu, logvar) BEFORE modality dropout.
        """
        cond_hm = self._cond_emb(K, PI_freq)
        hm_c = torch.cat([self._encode_heatmap_feat(heatmap, cond_hm), cond_hm], dim=1)
        _, _, joint_feat = self._encode_layout_joint(occupancy, impedance)
        layout_c = torch.cat([joint_feat, self.k_embedding(K)], dim=1)   # occupancy + impedance experts

        hm_mu = self.heatmap_mu(hm_c)
        hm_lv = self.heatmap_logvar(hm_c).clamp(EXPERT_LOGVAR_MIN, EXPERT_LOGVAR_MAX)
        occ_mu = self.occupancy_mu(layout_c)
        occ_lv = self.occupancy_logvar(layout_c).clamp(EXPERT_LOGVAR_MIN, EXPERT_LOGVAR_MAX)
        imp_mu = self.impedance_mu(layout_c)
        imp_lv = self.impedance_logvar(layout_c).clamp(EXPERT_LOGVAR_MIN, EXPERT_LOGVAR_MAX)

        expert_stats = {
            "heatmap": (hm_mu, hm_lv),
            "occupancy": (occ_mu, occ_lv),
            "impedance": (imp_mu, imp_lv),
        }
        mu_list, logvar_list = [hm_mu, occ_mu, imp_mu], [hm_lv, occ_lv, imp_lv]

        # Modality dropout: each expert is replaced by the prior with probability
        # `modality_dropout` (at least one kept; the heatmap expert is pinned on once
        # `modality_dropout_protect_heatmap` is set). It stops the encoder copying the input
        # heatmap straight through to the reconstruction.
        if self.training and self.modality_dropout > 0:
            dev = hm_mu.device
            keep = torch.bernoulli(torch.full((3,), 1.0 - self.modality_dropout, device=dev)).bool()
            if self.modality_dropout_protect_heatmap:
                keep = keep.clone()
                keep[0] = True
            if not keep.any():
                j = torch.randint(3, (1,), device=dev)
                keep = keep | (torch.arange(3, device=dev) == j)
            k3 = keep.view(3, 1, 1)
            mu_s = torch.where(k3, torch.stack(mu_list), torch.zeros_like(hm_mu).unsqueeze(0))
            lv_s = torch.where(k3, torch.stack(logvar_list), torch.zeros_like(hm_lv).unsqueeze(0))
            mu_list, logvar_list = list(mu_s.unbind(0)), list(lv_s.unbind(0))

        mu, logvar = self.product_of_experts(mu_list, logvar_list)
        logvar = torch.clamp(logvar, min=POSTERIOR_LOGVAR_MIN, max=POSTERIOR_LOGVAR_MAX)
        return self._reparameterize(mu, logvar), mu, logvar, expert_stats

    # ── Decode ───────────────────────────────────────────────────────────────

    def decode(self, z, K, PI_freq, occupancy: torch.Tensor | None = None):
        """z -> (heatmap (B,1,64,64), occupancy logits (B,52,T), impedance (B,1,231)).

        With ``occupancy`` given it is first hardened to exactly K decaps (the layout CAD would
        build; no gradient flows through the hardening), and that hard layout conditions the
        heatmap and impedance decoders. Without it the impedance decoder is conditioned on the
        decoded occupancy probabilities and the heatmap decoder gets no layout input.
        """
        k_emb = self.k_embedding(K)
        cond = self._cond_emb(K, PI_freq)

        occ_for_encoder = occ_for_hm = occupancy
        if occupancy is not None:
            if occupancy.dim() == 3:
                occ_for_encoder = hard_onehot_from_probs(occupancy, K)
                occ_for_hm = presence_from_onehot(occ_for_encoder)
            else:
                occ_for_hm = hard_presence(occupancy, K)
                occ_for_encoder = occ_for_hm

        hm_recon = self._decode_heatmap_tensor(z, cond, occupancy=occ_for_hm)
        occ_recon = self.occupancy_decoder(torch.cat([z, k_emb], dim=1))
        occ_for_imp = occ_for_encoder if occ_for_encoder is not None else torch.sigmoid(occ_recon)
        occ_ctx = self.occupancy_encoder(occ_for_imp)
        imp_recon = self.impedance_decoder(torch.cat([z, k_emb, occ_ctx], dim=1)).unsqueeze(1)
        return hm_recon, occ_recon, imp_recon

    def forward(self, heatmap, occupancy, impedance, K, PI_freq):
        z, mu, logvar, expert_stats = self.encode(heatmap, occupancy, impedance, K, PI_freq)
        hm_recon, occ_recon, imp_recon = self.decode(z, K, PI_freq, occupancy=occupancy)
        return hm_recon, occ_recon, imp_recon, mu, logvar, expert_stats
