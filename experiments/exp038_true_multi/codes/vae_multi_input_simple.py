import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Literal

from experiments.exp038_true_multi.codes.freq_conditioning import FiLM2d, FreqConditioner

Dr_value        = 0.1   # Global dropout rate
InferenceMode   = Literal["marginal", "layout", "anchor_blend", "encode"]
IMP_DROPOUT     = 0.15  # Dropout in impedance decoder MLP


class DeepMLP(nn.Module):
    """Deep MLP with LayerNorm, SiLU activations, and optional dropout."""

    def __init__(self, dims: list, dropout: float = 0.0):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.SiLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttn2d(nn.Module):
    """Lightweight spatial self-attention for 2D feature maps (B, C, H, W)."""
    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        seq = self.norm(x).view(B, C, H * W).permute(0, 2, 1)
        out, _ = self.attn(seq, seq, seq)
        return x + out.permute(0, 2, 1).view(B, C, H, W)


class MultiInputVAE(nn.Module):
    """
    Multi-modal VAE with a shared latent space + heatmap-private tail dims.

    - Heatmap expert predicts the full latent vector (shared + private).
    - Occupancy/impedance experts predict only the shared part; the private
      dims are forced to the unit-Gaussian prior (mu=0, logvar=0).
    - Decoder wiring:
        * Heatmap decoder sees full z (shared + private).
        * Occupancy/impedance decoders see only shared z.

    This implements "heatmap private dimensions" while keeping the overall
    latent size constant (e.g. total=32, heatmap_private=8, shared=24).
    """
    def __init__(
        self,
        latent_dim: int = 42,
        modality_dropout: float = 0.5,
        cond_dim: int = 8,
        heatmap_private_dim: int = 8,
        freq_fourier_features: int = 8,
        use_heatmap_film: bool = True,
    ):
        super().__init__()
        if heatmap_private_dim < 0:
            raise ValueError(f"heatmap_private_dim must be >= 0, got {heatmap_private_dim}")
        if heatmap_private_dim >= latent_dim:
            raise ValueError(
                f"heatmap_private_dim must be < latent_dim (total). Got heatmap_private_dim={heatmap_private_dim}, latent_dim={latent_dim}"
            )

        self.latent_dim          = latent_dim
        self.heatmap_private_dim = heatmap_private_dim
        self.shared_latent_dim   = latent_dim - heatmap_private_dim
        self.modality_dropout = modality_dropout
        self.modality_dropout_protect_heatmap = False
        self.cond_dim         = cond_dim
        self.use_heatmap_film = use_heatmap_film

        # K conditioning: K = number of 1s in occupancy vector, K ∈ {0, ..., 52}
        self.k_embedding = nn.Embedding(53, cond_dim)

        # PI_freq: Fourier features + MLP → cond_dim (stronger than single Linear)
        self.freq_conditioner = FreqConditioner(cond_dim, num_fourier=freq_fourier_features)
        # Legacy alias for old checkpoints (freq_proj.*)
        self.freq_proj = self.freq_conditioner

        # ── Encoders ─────────────────────────────────────────────────────────
        self.heatmap_enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(),
            nn.Dropout2d(Dr_value),
        )
        self.heatmap_enc_attn = SelfAttn2d(32, num_heads=1)
        self.heatmap_enc_film = FiLM2d(32, 2 * cond_dim) if use_heatmap_film else None
        self.heatmap_enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Dropout(Dr_value),
            nn.Linear(512, 64),
        )

        self.occupancy_encoder = nn.Sequential(
            nn.Linear(52, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Dropout(Dr_value),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Dropout(Dr_value),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
        )

        # Impedance input: (B, 1, 231) → flatten to 231 (log-z-score PI spectrum)
        self.impedance_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(231, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(Dr_value),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
        )

        # ── Latent projections (PoE experts) ────────────────────────────────
        # Heatmap: K + PI_freq.  Occupancy + impedance: K only (spectrum/layout, not PI frequency).
        enc_in_hm  = 64 + 2 * cond_dim
        enc_in_k   = 64 + cond_dim
        self.heatmap_mu       = nn.Linear(enc_in_hm, latent_dim)
        self.heatmap_logvar   = nn.Linear(enc_in_hm, latent_dim)
        self.occupancy_mu     = nn.Linear(enc_in_k, self.shared_latent_dim)
        self.occupancy_logvar = nn.Linear(enc_in_k, self.shared_latent_dim)
        self.impedance_mu     = nn.Linear(enc_in_k, self.shared_latent_dim)
        self.impedance_logvar = nn.Linear(enc_in_k, self.shared_latent_dim)

        # ── Decoders ────────────────────────────────────────────────────────
        dec_in_heatmap = latent_dim + 2 * cond_dim   # z + K + PI_freq → heatmap
        dec_in_occ     = self.shared_latent_dim + cond_dim
        dec_in_imp     = self.shared_latent_dim + cond_dim   # z_shared + K → impedance

        self.heatmap_fc = nn.Sequential(
            nn.Linear(dec_in_heatmap, 512),
            nn.LeakyReLU(),
            nn.Dropout(Dr_value),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Dropout(Dr_value),
            nn.Linear(512, 128 * 4 * 4),
            nn.LayerNorm(128 * 4 * 4),
            nn.LeakyReLU(),
        )
        self.heatmap_dec_deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(),
            nn.Dropout2d(Dr_value),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Dropout2d(Dr_value),
        )
        self.heatmap_dec_attn = SelfAttn2d(32, num_heads=1)
        self.heatmap_dec_film = FiLM2d(32, 2 * cond_dim) if use_heatmap_film else None
        self.heatmap_dec_deconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Dropout2d(Dr_value),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
        )

        self.occupancy_decoder = nn.Sequential(
            nn.Linear(dec_in_occ, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Dropout(Dr_value),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Dropout(Dr_value),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Dropout(Dr_value),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Dropout(Dr_value),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 52),
            # No Sigmoid here — BCEWithLogitsLoss is used in training (numerically stable)
            # and sigmoid is applied in inference() before returning probabilities
        )

        self.impedance_decoder = DeepMLP(
            [dec_in_imp, 256, 512, 1024, 512, 256, 231], dropout=IMP_DROPOUT
        )

    def _pad_shared_to_full(self, mu_shared: torch.Tensor, logvar_shared: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad shared (B, shared_latent_dim) expert stats to full (B, latent_dim).

        Private dims are set to unit Gaussian prior: mu=0, logvar=0.
        """
        if self.heatmap_private_dim == 0:
            return mu_shared, logvar_shared
        B = mu_shared.shape[0]
        zeros_mu = torch.zeros(B, self.heatmap_private_dim, device=mu_shared.device, dtype=mu_shared.dtype)
        zeros_lv = torch.zeros(B, self.heatmap_private_dim, device=logvar_shared.device, dtype=logvar_shared.dtype)
        mu_full = torch.cat([mu_shared, zeros_mu], dim=1)
        lv_full = torch.cat([logvar_shared, zeros_lv], dim=1)
        return mu_full, lv_full

    # ── PoE fusion ────────────────────────────────────────────────────────────

    def product_of_experts(self, mu_list, logvar_list, eps=1e-8):
        """
        Combine Gaussian experts via Product of Experts.
        A unit-Gaussian prior (mu=0, logvar=0) is always appended as an anchor.
        """
        prior_mu     = torch.zeros_like(mu_list[0])
        prior_logvar = torch.zeros_like(logvar_list[0])
        mu_list      = mu_list      + [prior_mu]
        logvar_list  = logvar_list  + [prior_logvar]

        var_list           = [torch.exp(lv) + eps for lv in logvar_list]
        precision_list     = [1.0 / v for v in var_list]
        combined_precision = torch.stack(precision_list).sum(dim=0)
        combined_var       = 1.0 / (combined_precision + eps)
        combined_mu        = combined_var * torch.stack(
            [mu * p for mu, p in zip(mu_list, precision_list)]
        ).sum(dim=0)
        return combined_mu, torch.log(combined_var + eps)

    # ── Core encode / decode / reparameterize ─────────────────────────────────

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * torch.clamp(logvar, min=-4.0, max=2.0))
        return mu + torch.randn_like(std) * std

    def _cond_emb(self, K: torch.Tensor, PI_freq: torch.Tensor) -> torch.Tensor:
        """Combine K embedding (discrete) and PI_freq projection (continuous).

        Args:
            K:       (B,) LongTensor  — number of active decaps (0–52)
            PI_freq: (B,) FloatTensor — log10-normalised frequency in [0, 1]
        Returns:
            (B, 2*cond_dim) combined conditioning vector
        """
        k_emb = self.k_embedding(K)
        f_emb = self.freq_conditioner(PI_freq)
        return torch.cat([k_emb, f_emb], dim=1)

    def encode(self, heatmap, occupancy, impedance, K, PI_freq):
        """
        Encode all three modalities, fuse via PoE, return z + stats.

        Args:
            heatmap:   (B, 1, 64, 64)
            occupancy: (B, 52)
            impedance: (B, 1, 231)
            K:         (B,) LongTensor
            PI_freq:   (B,) FloatTensor  — log10-norm freq in [0, 1]
        Returns:
            z          (B, latent_dim)
            mu         (B, latent_dim)
            logvar     (B, latent_dim)
            expert_stats  dict
        """
        cond_hm = self._cond_emb(K, PI_freq)
        h1 = self.heatmap_enc1(heatmap)
        h1 = self.heatmap_enc_attn(h1)
        if self.heatmap_enc_film is not None:
            h1 = self.heatmap_enc_film(h1, cond_hm)
        heatmap_feat = self.heatmap_enc2(h1)
        occupancy_feat = self.occupancy_encoder(occupancy)
        impedance_feat = self.impedance_encoder(impedance)

        k_emb  = self.k_embedding(K)
        hm_c   = torch.cat([heatmap_feat, cond_hm], dim=1)
        occ_c  = torch.cat([occupancy_feat, k_emb], dim=1)
        imp_c  = torch.cat([impedance_feat, k_emb], dim=1)

        # Per-modality expert predictions (pre-PoE, pre-dropout)
        # Clamp expert logvars before PoE to prevent variance explosion.
        # Without this, an encoder can learn exp(logvar)→∞ → precision≈0 →
        # near-zero PoE gradient → unbounded runaway.
        _EXP_LV_MIN, _EXP_LV_MAX = -6.0, 1.0   # cap expert σ ≤ 1.65 to prevent experts opting out of PoE fusion
        hm_mu = self.heatmap_mu(hm_c)
        hm_lv = self.heatmap_logvar(hm_c).clamp(_EXP_LV_MIN, _EXP_LV_MAX)

        occ_mu_s = self.occupancy_mu(occ_c)
        occ_lv_s = self.occupancy_logvar(occ_c).clamp(_EXP_LV_MIN, _EXP_LV_MAX)
        occ_mu, occ_lv = self._pad_shared_to_full(occ_mu_s, occ_lv_s)

        imp_mu_s = self.impedance_mu(imp_c)
        imp_lv_s = self.impedance_logvar(imp_c).clamp(_EXP_LV_MIN, _EXP_LV_MAX)
        imp_mu, imp_lv = self._pad_shared_to_full(imp_mu_s, imp_lv_s)

        # Expert stats before dropout (for monitoring only — not used in loss)
        expert_stats = {
            'heatmap':   (hm_mu,  hm_lv),
            'occupancy': (occ_mu, occ_lv),
            'impedance': (imp_mu, imp_lv),
        }

        mu_list     = [hm_mu,  occ_mu,  imp_mu]
        logvar_list = [hm_lv,  occ_lv,  imp_lv]

        # Modality dropout: replace dropped experts with prior (keep ≥1); no .item() for compile
        if self.training and self.modality_dropout > 0:
            dev = hm_mu.device
            keep = torch.bernoulli(
                torch.full((3,), 1.0 - self.modality_dropout, device=dev),
            ).bool()
            if self.modality_dropout_protect_heatmap:
                keep = keep.clone()
                keep[0] = True
            if not keep.any():
                j = torch.randint(3, (1,), device=dev)
                keep = keep | (torch.arange(3, device=dev) == j)
            prior_mu = torch.zeros_like(mu_list[0])
            prior_lv = torch.zeros_like(logvar_list[0])
            mu_s = torch.stack(mu_list)
            lv_s = torch.stack(logvar_list)
            k3 = keep.view(3, 1, 1)
            mu_s = torch.where(k3, mu_s, prior_mu.unsqueeze(0))
            lv_s = torch.where(k3, lv_s, prior_lv.unsqueeze(0))
            mu_list = list(mu_s.unbind(0))
            logvar_list = list(lv_s.unbind(0))

        mu, logvar = self.product_of_experts(mu_list, logvar_list)
        logvar     = torch.clamp(logvar, min=-4.0, max=2.0)
        z          = self._reparameterize(mu, logvar)

        return z, mu, logvar, expert_stats

    def decode(self, z, K, PI_freq):
        k_emb    = self.k_embedding(K)
        cond     = self._cond_emb(K, PI_freq)
        z_shared = z[:, : self.shared_latent_dim]
        dec_hm   = torch.cat([z, cond], dim=1)
        dec_occ  = torch.cat([z_shared, k_emb], dim=1)
        dec_imp  = torch.cat([z_shared, k_emb], dim=1)

        hm_feat  = self.heatmap_fc(dec_hm)
        hm_feat  = self.heatmap_dec_deconv1(hm_feat.view(-1, 128, 4, 4))
        hm_feat  = self.heatmap_dec_attn(hm_feat)
        if self.heatmap_dec_film is not None:
            hm_feat = self.heatmap_dec_film(hm_feat, cond)
        hm_recon = self.heatmap_dec_deconv2(hm_feat)

        occ_recon = self.occupancy_decoder(dec_occ)
        imp_recon = self.impedance_decoder(dec_imp).unsqueeze(1)                     # (B, 1, 231)

        return hm_recon, occ_recon, imp_recon

    def decode_heatmap_blended(
        self,
        z: torch.Tensor,
        K: torch.Tensor,
        mhz: float | torch.Tensor,
        *,
        pi_freq_unit: str = "mhz",
    ) -> torch.Tensor:
        """Decode heatmap at arbitrary MHz by log-blending decodes at bracketing training anchors.

        Improves off-anchor frequencies (e.g. 80, 250 MHz) that were never in the dataset.
        """
        from experiments.exp038_true_multi.codes.freq_inference_utils import (
            bracket_anchors_mhz,
            is_training_anchor,
            pi_norm_tensor,
        )
        from src_vae.others.pi_freq_utils import pi_freq_mhz_to_norm

        if isinstance(mhz, torch.Tensor) and mhz.numel() > 1:
            raise ValueError("decode_heatmap_blended expects a scalar MHz per call")
        mhz_f = float(mhz.item() if isinstance(mhz, torch.Tensor) else mhz)
        B = z.shape[0]
        device = z.device

        if is_training_anchor(mhz_f):
            pi_t = pi_freq_mhz_to_norm(mhz_f) if pi_freq_unit == "mhz" else float(mhz_f)
            pi = torch.full((B,), pi_t, dtype=torch.float32, device=device)
            return self.decode(z, K, pi)[0]

        lo, hi, t = bracket_anchors_mhz(mhz_f)
        pi_lo = pi_norm_tensor(lo, B, device)
        pi_hi = pi_norm_tensor(hi, B, device)
        hm_lo = self.decode(z, K, pi_lo)[0]
        hm_hi = self.decode(z, K, pi_hi)[0]
        if lo == hi:
            return hm_lo
        w = torch.tensor(t, dtype=hm_lo.dtype, device=device).view(1, 1, 1, 1)
        return (1.0 - w) * hm_lo + w * hm_hi

    def decode_impedance(self, z: torch.Tensor, K, PI_freq=None) -> torch.Tensor:
        """Decode only impedance from latent z and K (PI_freq ignored — heatmap-only cond).

        Useful for impedance-guided latent optimization (faster than full decode).

        Args:
            z:       (B, latent_dim) or (latent_dim,)
            K:       int or LongTensor broadcastable to (B,)
            PI_freq: unused (kept for call-site compatibility)

        Returns:
            imp_recon: (B, 1, 231) (or (1,231) if input z was 1D)
        """
        squeeze = False
        if z.dim() == 1:
            z = z.unsqueeze(0)
            squeeze = True
        if z.dim() != 2:
            raise ValueError(f"Expected z shape (B, latent_dim) or (latent_dim,), got {tuple(z.shape)}")

        B = z.shape[0]
        if isinstance(K, int):
            K_tensor = torch.full((B,), K, dtype=torch.long, device=z.device)
        elif isinstance(K, torch.Tensor):
            K_tensor = K.to(device=z.device, dtype=torch.long)
            if K_tensor.dim() == 0:
                K_tensor = K_tensor.expand(B)
            else:
                K_tensor = K_tensor.view(-1)
            if K_tensor.shape[0] != B:
                raise ValueError(f"K has shape {tuple(K_tensor.shape)} but z batch size is {B}")
        else:
            raise TypeError(f"K must be int or torch.Tensor, got {type(K).__name__}")

        k_emb     = self.k_embedding(K_tensor)
        z_shared  = z[:, : self.shared_latent_dim]
        dec_imp   = torch.cat([z_shared, k_emb], dim=1)
        imp_recon = self.impedance_decoder(dec_imp).unsqueeze(1)                     # (B, 1, 231)
        return imp_recon[0] if squeeze else imp_recon

    def spline_l1(self) -> torch.Tensor:
        """Returns zero (no KAN spline weights; kept for API compatibility)."""
        return torch.tensor(0.0)

    def _encode_impedance_expert(self, impedance, K: torch.Tensor):
        """Shared helper: impedance tensor → (mu_full, lv_full); conditioned on K only."""
        imp_in = impedance
        if imp_in.dim() == 1:
            imp_in = imp_in.unsqueeze(0).unsqueeze(1)   # (1, 1, 231)
        elif imp_in.dim() == 2:
            if imp_in.shape[1] == 231:
                imp_in = imp_in.unsqueeze(1)            # (B, 1, 231)
            elif imp_in.shape[1] % 231 == 0:
                imp_in = imp_in.reshape(-1, 1, 231)       # legacy multi-ch → ch0 only
            else:
                raise ValueError(
                    f"Expected impedance (B,231) or (B,1,231); got {tuple(imp_in.shape)}"
                )
        elif imp_in.dim() == 3:
            if imp_in.shape[2] != 231:
                raise ValueError(f"Expected last dim 231; got {tuple(imp_in.shape)}")
            imp_in = imp_in[:, :1, :]                     # ch0 only (legacy 2/3-ch)
        else:
            raise ValueError(f"Expected impedance (B,1,231) or (B,231); got {tuple(imp_in.shape)}")
        feat   = self.impedance_encoder(imp_in)
        k_emb  = self.k_embedding(K)
        feat_c = torch.cat([feat, k_emb], dim=1)
        mu_s   = self.impedance_mu(feat_c)
        lv_s   = self.impedance_logvar(feat_c).clamp(-6.0, 1.0)
        return self._pad_shared_to_full(mu_s, lv_s)

    def encode_cross_modal(self, source: str, K, PI_freq,
                           heatmap=None, occupancy=None, impedance=None):
        """
        Encode from one or two modalities via PoE (missing modalities → prior).

        source:
          'heatmap'   — heatmap only               (heatmap required)
          'impedance' — impedance only              (impedance required)
          'occ_imp'   — occupancy + impedance       (occupancy + impedance required)
                        → useful for predicting the heatmap from measurements

        Returns z (B, latent_dim).
        """
        cond = self._cond_emb(K, PI_freq)
        mu_list, lv_list = [], []

        if source == 'heatmap':
            if heatmap is None:
                raise ValueError("heatmap must be provided when source='heatmap'")
            feat   = self.heatmap_enc2(self.heatmap_enc_attn(self.heatmap_enc1(heatmap)))
            feat_c = torch.cat([feat, cond], dim=1)
            mu_list.append(self.heatmap_mu(feat_c))
            lv_list.append(self.heatmap_logvar(feat_c).clamp(-6.0, 1.0))

        elif source == 'impedance':
            if impedance is None:
                raise ValueError("impedance must be provided when source='impedance'")
            mu_s, lv_s = self._encode_impedance_expert(impedance, K)
            mu_list.append(mu_s); lv_list.append(lv_s)

        elif source == 'occ_imp':
            if occupancy is None or impedance is None:
                raise ValueError("occupancy and impedance must both be provided when source='occ_imp'")
            k_emb    = self.k_embedding(K)
            # Occupancy expert
            occ_c    = torch.cat([self.occupancy_encoder(occupancy), k_emb], dim=1)
            occ_mu_s = self.occupancy_mu(occ_c)
            occ_lv_s = self.occupancy_logvar(occ_c).clamp(-6.0, 1.0)
            occ_mu, occ_lv = self._pad_shared_to_full(occ_mu_s, occ_lv_s)
            mu_list.append(occ_mu); lv_list.append(occ_lv)
            # Impedance expert
            imp_mu, imp_lv = self._encode_impedance_expert(impedance, K)
            mu_list.append(imp_mu); lv_list.append(imp_lv)

        else:
            raise ValueError(
                f"Unsupported source: {source!r}. Use 'heatmap', 'impedance', or 'occ_imp'."
            )

        mu, logvar = self.product_of_experts(mu_list, lv_list)
        z = self._reparameterize(mu, torch.clamp(logvar, -4.0, 2.0))
        return z

    def forward(self, heatmap, occupancy, impedance, K, PI_freq):
        z, mu, logvar, expert_stats = self.encode(heatmap, occupancy, impedance, K, PI_freq)
        hm_recon, occ_recon, imp_recon = self.decode(z, K, PI_freq)
        return hm_recon, occ_recon, imp_recon, mu, logvar, expert_stats

    def _sample_z_marginal(
        self,
        num_samples: int,
        device: torch.device,
        K: int,
        *,
        latent_stats=None,
        per_K_latent_stats=None,
        shared_temp: float = 1.0,
    ) -> torch.Tensor:
        stats = latent_stats if latent_stats is not None else {}
        if per_K_latent_stats is not None:
            k_key = str(K)
            if k_key in per_K_latent_stats:
                stats = per_K_latent_stats[k_key]
            else:
                available = [int(k) for k in per_K_latent_stats]
                if available:
                    nearest = str(min(available, key=lambda x: abs(x - K)))
                    stats = per_K_latent_stats[nearest]
        s = stats.get("latent", {})
        if "mu_mean_per_dim" in s and "agg_std_per_dim" in s:
            mu = torch.tensor(s["mu_mean_per_dim"], dtype=torch.float32, device=device)
            agg_std = torch.tensor(s["agg_std_per_dim"], dtype=torch.float32, device=device)
            return torch.randn(num_samples, self.latent_dim, device=device) * (agg_std * shared_temp) + mu
        mu_val = s.get("mu_mean", 0.0)
        agg_std = (s.get("mu_std", 1.0) ** 2 + s.get("sigma_mean", 1.0) ** 2) ** 0.5
        return torch.randn(num_samples, self.latent_dim, device=device) * (agg_std * shared_temp) + mu_val

    def encode_layout_latent(
        self,
        occupancy: torch.Tensor,
        impedance: torch.Tensor,
        K: torch.Tensor,
        PI_freq: torch.Tensor,
    ) -> torch.Tensor:
        """Latent from occupancy + impedance experts (layout), conditioned on PI_freq at decode."""
        return self.encode_cross_modal(
            "occ_imp", K, PI_freq, occupancy=occupancy, impedance=impedance,
        )

    def inference(
        self,
        num_samples,
        device,
        K,
        PI_freq: "float | torch.Tensor" = 200.0,
        pi_freq_unit: str = "mhz",
        latent_stats=None,
        per_K_latent_stats=None,
        shared_temp: float = 1.0,
        mode: InferenceMode = "layout",
        pi_ref_mhz: float = 200.0,
        occupancy: torch.Tensor | None = None,
        impedance: torch.Tensor | None = None,
        heatmap: torch.Tensor | None = None,
    ):
        """
        Generate samples with PI_freq at decode (and optionally in the latent).

        Args:
            mode:
                ``marginal`` — z ~ global stats (legacy; PI_freq decode only).
                ``layout``   — z from occ+imp PoE (marginal occ/imp if not passed).
                ``anchor_blend`` — same z as layout; heatmap = blend of decodes at bracketing anchors.
                ``encode``   — full encode(hm, occ, imp, K, PI_freq); requires heatmap.
            pi_ref_mhz: reference MHz for marginal occ/imp before layout encode (layout mode).
            occupancy, impedance: optional (B,52) / (B,1,231) for layout mode.
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            if isinstance(K, int):
                K_tensor = torch.full((num_samples,), K, dtype=torch.long, device=device)
            else:
                K_tensor = K.long().to(device)
                if K_tensor.dim() == 0:
                    K_tensor = K_tensor.expand(num_samples)

            from src_vae.others.pi_freq_utils import pi_freq_norm_for_model
            PI_freq_t = pi_freq_norm_for_model(
                PI_freq, num_samples, unit=pi_freq_unit, device=device,
            )
            k_int = int(K_tensor[0].item()) if K_tensor.numel() else int(K)

            if mode == "encode":
                if heatmap is None or occupancy is None or impedance is None:
                    raise ValueError("mode='encode' requires heatmap, occupancy, and impedance")
                z, _, _, _ = self.encode(heatmap, occupancy, impedance, K_tensor, PI_freq_t)
            elif mode == "marginal":
                z = self._sample_z_marginal(
                    num_samples, device, k_int,
                    latent_stats=latent_stats,
                    per_K_latent_stats=per_K_latent_stats,
                    shared_temp=shared_temp,
                )
            elif mode in ("layout", "anchor_blend"):
                pi_ref = pi_freq_norm_for_model(
                    pi_ref_mhz, num_samples, unit="mhz", device=device,
                )
                if occupancy is None or impedance is None:
                    z0 = self._sample_z_marginal(
                        num_samples, device, k_int,
                        latent_stats=latent_stats,
                        per_K_latent_stats=per_K_latent_stats,
                        shared_temp=shared_temp,
                    )
                    _, occ_logits, imp0 = self.decode(z0, K_tensor, pi_ref)
                    occupancy = torch.sigmoid(occ_logits)
                    impedance = imp0
                # Encode layout at pi_ref (matches training: same PI_freq for occ/imp encode + decode).
                # Do not condition the encoder on the target sweep MHz — only the decoder sees that.
                z = self.encode_layout_latent(occupancy, impedance, K_tensor, pi_ref)
            else:
                raise ValueError(f"Unknown inference mode: {mode!r}")

            if mode == "anchor_blend":
                if pi_freq_unit == "mhz":
                    if isinstance(PI_freq, torch.Tensor):
                        mhz_scalar = float(PI_freq.reshape(-1)[0].item())
                    else:
                        mhz_scalar = float(PI_freq)
                else:
                    mhz_scalar = 200.0
                heatmap = self.decode_heatmap_blended(z, K_tensor, mhz_scalar)
                _, occupancy_out, impedance = self.decode(z, K_tensor, PI_freq_t)
            else:
                heatmap, occupancy_out, impedance = self.decode(z, K_tensor, PI_freq_t)
            occupancy_out = torch.sigmoid(occupancy_out)

        if was_training:
            self.train()
        return heatmap, occupancy_out, impedance
