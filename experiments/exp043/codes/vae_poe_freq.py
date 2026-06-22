"""Multi-input VAE with PI_freq Product-of-Experts expert (exp043).

Purpose:
    Extend ``MultiInputVAE`` with a frequency-only PoE expert fused with layout experts for multifreq training.

Run:
    Import only — instantiated by ``train_vae_simple``, ``inference_vae``, ``exp043_eval_common``.

Agent notes:
    - What: Core exp043 model class ``MultiInputVAEPoeFreq`` (shared latent + optional private heatmap dims).
    - Usage: Import and construct with ``latent_dim``, ``cond_dim``, ``use_freq_poe_expert``; not run directly.
    - Key symbols: ``MultiInputVAEPoeFreq``, ``encode_layout_latent``, ``encode_cross_modal``
    - Flag: ``use_freq_poe_expert`` (default True) enables MHz-conditioned PoE fusion.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from experiments.exp043.codes.vae_multi_input_simple import MultiInputVAE

_EXP_LV_MIN, _EXP_LV_MAX = -6.0, 1.0


class MultiInputVAEPoeFreq(MultiInputVAE):
    """PoE fusion: layout experts + optional PI_freq expert on full or private latent dims."""

    def __init__(
        self,
        latent_dim: int = 42,
        modality_dropout: float = 0.5,
        cond_dim: int = 8,
        heatmap_private_dim: int = 0,
        freq_fourier_features: int = 8,
        use_heatmap_film: bool = True,
        *,
        use_freq_poe_expert: bool = True,
        use_heatmap_unet_skips: bool = True,
        use_occ_spatial_decoder: bool = True,
    ):
        if heatmap_private_dim < 0:
            raise ValueError(f"heatmap_private_dim must be >= 0, got {heatmap_private_dim}")
        super().__init__(
            latent_dim=latent_dim,
            modality_dropout=modality_dropout,
            cond_dim=cond_dim,
            heatmap_private_dim=heatmap_private_dim,
            freq_fourier_features=freq_fourier_features,
            use_heatmap_film=use_heatmap_film,
            use_heatmap_unet_skips=use_heatmap_unet_skips,
            use_occ_spatial_decoder=use_occ_spatial_decoder,
        )
        self.use_freq_poe_expert = use_freq_poe_expert
        self._freq_expert_dim = latent_dim if heatmap_private_dim == 0 else heatmap_private_dim
        self.freq_poe_mu = nn.Linear(cond_dim, self._freq_expert_dim)
        self.freq_poe_logvar = nn.Linear(cond_dim, self._freq_expert_dim)

    def _encode_freq_poe_expert(
        self, PI_freq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """PI_freq-only Gaussian expert (full latent or private tail)."""
        f_emb = self.freq_conditioner(PI_freq)
        mu_p = self.freq_poe_mu(f_emb)
        lv_p = self.freq_poe_logvar(f_emb).clamp(_EXP_LV_MIN, _EXP_LV_MAX)
        return mu_p, lv_p

    def _pad_freq_expert_to_full(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map freq expert stats to full ``latent_dim`` for PoE fusion."""
        if self.heatmap_private_dim == 0:
            return mu, logvar
        b = mu.shape[0]
        dev, dt = mu.device, mu.dtype
        z_shared = torch.zeros(b, self.shared_latent_dim, device=dev, dtype=dt)
        lv_shared = torch.zeros(b, self.shared_latent_dim, device=dev, dtype=dt)
        mu_full = torch.cat([z_shared, mu], dim=1)
        lv_full = torch.cat([lv_shared, logvar], dim=1)
        return mu_full, lv_full

    def _append_freq_poe_expert(
        self,
        mu_list: list[torch.Tensor],
        lv_list: list[torch.Tensor],
        PI_freq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.use_freq_poe_expert:
            return None
        mu_p, lv_p = self._encode_freq_poe_expert(PI_freq)
        mu_f, lv_f = self._pad_freq_expert_to_full(mu_p, lv_p)
        mu_list.append(mu_f)
        lv_list.append(lv_f)
        return mu_f, lv_f

    def encode(self, heatmap, occupancy, impedance, K, PI_freq):
        cond_hm = self._cond_emb(K, PI_freq)
        hm_skips = None
        if self.use_heatmap_unet_skips:
            heatmap_feat, hm_skips = self._encode_heatmap_feat(heatmap, cond_hm, return_skips=True)
        else:
            heatmap_feat = self._encode_heatmap_feat(heatmap, cond_hm)
        occupancy_feat = self.occupancy_encoder(occupancy)
        impedance_feat = self.impedance_encoder(impedance)

        k_emb = self.k_embedding(K)
        hm_c = torch.cat([heatmap_feat, cond_hm], dim=1)
        occ_c = torch.cat([occupancy_feat, k_emb], dim=1)
        imp_c = torch.cat([impedance_feat, k_emb], dim=1)

        hm_mu = self.heatmap_mu(hm_c)
        hm_lv = self.heatmap_logvar(hm_c).clamp(_EXP_LV_MIN, _EXP_LV_MAX)

        occ_mu_s = self.occupancy_mu(occ_c)
        occ_lv_s = self.occupancy_logvar(occ_c).clamp(_EXP_LV_MIN, _EXP_LV_MAX)
        occ_mu, occ_lv = self._pad_shared_to_full(occ_mu_s, occ_lv_s)

        imp_mu_s = self.impedance_mu(imp_c)
        imp_lv_s = self.impedance_logvar(imp_c).clamp(_EXP_LV_MIN, _EXP_LV_MAX)
        imp_mu, imp_lv = self._pad_shared_to_full(imp_mu_s, imp_lv_s)

        freq_pair = self._encode_freq_poe_expert(PI_freq)
        if freq_pair is not None:
            freq_mu, freq_lv = self._pad_freq_expert_to_full(*freq_pair)
        else:
            freq_mu = freq_lv = None

        expert_stats = {
            "heatmap": (hm_mu, hm_lv),
            "occupancy": (occ_mu, occ_lv),
            "impedance": (imp_mu, imp_lv),
            "heatmap_skips": hm_skips,
        }
        if freq_mu is not None:
            expert_stats["freq_poe"] = (freq_mu, freq_lv)

        mu_list = [hm_mu, occ_mu, imp_mu]
        logvar_list = [hm_lv, occ_lv, imp_lv]

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

        self._append_freq_poe_expert(mu_list, logvar_list, PI_freq)

        mu, logvar = self.product_of_experts(mu_list, logvar_list)
        logvar = torch.clamp(logvar, min=-4.0, max=2.0)
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar, expert_stats

    def encode_cross_modal(
        self,
        source: str,
        K,
        PI_freq,
        heatmap=None,
        occupancy=None,
        impedance=None,
    ):
        cond = self._cond_emb(K, PI_freq)
        mu_list, lv_list = [], []

        if source == "heatmap":
            if heatmap is None:
                raise ValueError("heatmap must be provided when source='heatmap'")
            feat = self._encode_heatmap_feat(heatmap, cond)
            feat_c = torch.cat([feat, cond], dim=1)
            mu_list.append(self.heatmap_mu(feat_c))
            lv_list.append(self.heatmap_logvar(feat_c).clamp(_EXP_LV_MIN, _EXP_LV_MAX))

        elif source == "impedance":
            if impedance is None:
                raise ValueError("impedance must be provided when source='impedance'")
            imp_mu, imp_lv = self._encode_impedance_expert(impedance, K)
            mu_list.append(imp_mu)
            lv_list.append(imp_lv)

        elif source == "occ_imp":
            if occupancy is None or impedance is None:
                raise ValueError(
                    "occupancy and impedance must both be provided when source='occ_imp'",
                )
            k_emb = self.k_embedding(K)
            occ_c = torch.cat([self.occupancy_encoder(occupancy), k_emb], dim=1)
            occ_mu_s = self.occupancy_mu(occ_c)
            occ_lv_s = self.occupancy_logvar(occ_c).clamp(_EXP_LV_MIN, _EXP_LV_MAX)
            occ_mu, occ_lv = self._pad_shared_to_full(occ_mu_s, occ_lv_s)
            mu_list.append(occ_mu)
            lv_list.append(occ_lv)
            imp_mu, imp_lv = self._encode_impedance_expert(impedance, K)
            mu_list.append(imp_mu)
            lv_list.append(imp_lv)

        else:
            raise ValueError(
                f"Unsupported source: {source!r}. Use 'heatmap', 'impedance', or 'occ_imp'.",
            )

        self._append_freq_poe_expert(mu_list, lv_list, PI_freq)

        mu, logvar = self.product_of_experts(mu_list, lv_list)
        z = self._reparameterize(mu, torch.clamp(logvar, -4.0, 2.0))
        return z

    def encode_layout_latent(
        self,
        occupancy: torch.Tensor,
        impedance: torch.Tensor,
        K: torch.Tensor,
        PI_freq: torch.Tensor,
    ) -> torch.Tensor:
        """Layout z = PoE(occ, imp, PI_freq expert); decode still uses PI_freq FiLM."""
        return self.encode_cross_modal(
            "occ_imp", K, PI_freq, occupancy=occupancy, impedance=impedance,
        )
