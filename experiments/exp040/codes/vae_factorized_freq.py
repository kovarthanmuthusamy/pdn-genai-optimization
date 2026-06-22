"""
Factorized-frequency heatmap VAE (exp040).

Heatmap decode:
    H(x, y | z, K, MHz) = sum_k alpha_k(MHz) * B_k(x, y | z, K)  [+ optional residual]

- Spatial modes B_k depend on layout latent z and K only (not PI_freq).
- Mixing weights alpha_k depend on PI_freq only (shared across layouts).
- Occupancy / impedance paths unchanged from MultiInputVAE (K-only experts).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.exp038_true_multi.codes.freq_conditioning import FiLM2d
from experiments.exp038_true_multi.codes.vae_multi_input_simple import (
    Dr_value,
    MultiInputVAE,
    SelfAttn2d,
)

# Set during decode; read by exp040 training loss hook.
LAST_BASES: torch.Tensor | None = None


class FactorizedFreqVAE(MultiInputVAE):
  """Multifreq VAE with PI-frequency factorized heatmap decoder."""

  def __init__(
      self,
      latent_dim: int = 42,
      modality_dropout: float = 0.5,
      cond_dim: int = 8,
      heatmap_private_dim: int = 8,
      freq_fourier_features: int = 8,
      use_heatmap_film: bool = True,
      *,
      num_freq_modes: int = 6,
      freq_alpha_softmax: bool = False,
      use_freq_residual: bool = True,
      residual_gain_init: float = 0.15,
  ):
      super().__init__(
          latent_dim=latent_dim,
          modality_dropout=modality_dropout,
          cond_dim=cond_dim,
          heatmap_private_dim=heatmap_private_dim,
          freq_fourier_features=freq_fourier_features,
          use_heatmap_film=use_heatmap_film,
      )
      self.num_freq_modes = int(num_freq_modes)
      self.freq_alpha_softmax = bool(freq_alpha_softmax)
      self.use_freq_residual = bool(use_freq_residual)

      dec_in_basis = latent_dim + cond_dim
      self.basis_fc = nn.Sequential(
          nn.Linear(dec_in_basis, 512),
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

      self.basis_dec_deconv1 = nn.Sequential(
          nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
          nn.GroupNorm(8, 64),
          nn.LeakyReLU(),
          nn.Dropout2d(Dr_value),
          nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
          nn.LeakyReLU(),
          nn.Dropout2d(Dr_value),
      )
      self.basis_dec_attn = SelfAttn2d(32, num_heads=1)
      self.basis_dec_film = FiLM2d(32, cond_dim) if use_heatmap_film else None
      self.basis_dec_deconv2 = nn.Sequential(
          nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
          nn.LeakyReLU(),
          nn.Dropout2d(Dr_value),
          nn.ConvTranspose2d(16, self.num_freq_modes, kernel_size=4, stride=2, padding=1),
      )

      self.freq_alpha_head = nn.Linear(cond_dim, self.num_freq_modes)
      nn.init.zeros_(self.freq_alpha_head.bias)

      if self.use_freq_residual:
          self.residual_gain = nn.Parameter(
              torch.tensor(float(residual_gain_init), dtype=torch.float32),
          )
      else:
          self.residual_gain = None

      self._last_bases: torch.Tensor | None = None
      self._last_alpha: torch.Tensor | None = None
      # When True, skip parent heatmap decoder residual at inference (see inspect_sweep_artifacts).
      self.inference_factorized_only: bool = False

  def _decode_bases(self, z: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
      """Layout spatial modes B_k: (B, M, 64, 64)."""
      k_emb = self.k_embedding(K)
      dec_in = torch.cat([z, k_emb], dim=1)
      hm_feat = self.basis_fc(dec_in)
      hm_feat = self.basis_dec_deconv1(hm_feat.view(-1, 128, 4, 4))
      hm_feat = self.basis_dec_attn(hm_feat)
      if self.basis_dec_film is not None:
          hm_feat = self.basis_dec_film(hm_feat, k_emb)
      return self.basis_dec_deconv2(hm_feat)

  def _alpha_from_pi_freq(self, PI_freq: torch.Tensor) -> torch.Tensor:
      f_emb = self.freq_conditioner(PI_freq)
      alpha = self.freq_alpha_head(f_emb)
      if self.freq_alpha_softmax:
          alpha = F.softmax(alpha, dim=-1)
      return alpha

  def _mix_bases(self, bases: torch.Tensor, PI_freq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
      alpha = self._alpha_from_pi_freq(PI_freq)
      hm = torch.einsum("bm,bmhw->bhw", alpha, bases).unsqueeze(1)
      return hm, alpha

  def decode_heatmap_from_z(
      self,
      z: torch.Tensor,
      K: torch.Tensor,
      PI_freq: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
      bases = self._decode_bases(z, K)
      hm, alpha = self._mix_bases(bases, PI_freq)
      use_res = (
          self.use_freq_residual
          and self.residual_gain is not None
          and not self.inference_factorized_only
      )
      if use_res:
          hm_res, _, _ = super().decode(z, K, PI_freq)
          hm = hm + self.residual_gain * hm_res
      self._last_bases = bases
      self._last_alpha = alpha
      global LAST_BASES
      LAST_BASES = bases
      return hm, alpha, bases

  def decode(self, z: torch.Tensor, K: torch.Tensor, PI_freq: torch.Tensor):
      hm, _, _ = self.decode_heatmap_from_z(z, K, PI_freq)
      k_emb = self.k_embedding(K)
      z_shared = z[:, : self.shared_latent_dim]
      dec_occ = torch.cat([z_shared, k_emb], dim=1)
      dec_imp = torch.cat([z_shared, k_emb], dim=1)
      occ_recon = self.occupancy_decoder(dec_occ)
      imp_recon = self.impedance_decoder(dec_imp).unsqueeze(1)
      return hm, occ_recon, imp_recon

  def decode_heatmap_blended(
      self,
      z: torch.Tensor,
      K: torch.Tensor,
      mhz: float | torch.Tensor,
      *,
      pi_freq_unit: str = "mhz",
  ) -> torch.Tensor:
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


def modes_orthogonality_loss(bases: torch.Tensor) -> torch.Tensor:
      """Penalize correlated spatial modes (B, M, H, W)."""
      B, M, H, W = bases.shape
      x = bases.reshape(B, M, -1)
      x = F.normalize(x, dim=-1, eps=1e-6)
      gram = torch.bmm(x, x.transpose(1, 2))
      eye = torch.eye(M, device=bases.device, dtype=bases.dtype).unsqueeze(0)
      return F.mse_loss(gram, eye.expand(B, -1, -1))
