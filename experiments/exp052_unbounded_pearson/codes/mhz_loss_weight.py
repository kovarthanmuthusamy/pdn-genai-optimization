"""Per-sample heatmap loss multipliers vs PI frequency (MHz)."""

from __future__ import annotations

import math

import torch

from experiments.exp038_true_multi.codes.train_vae_simple import Config
from src_vae.others.pi_freq_utils import _LOG10_MIN, _LOG10_RANGE


def pi_norm_to_mhz(pi_norm: torch.Tensor) -> torch.Tensor:
    log10_hz = pi_norm.float() * _LOG10_RANGE + _LOG10_MIN
    return torch.pow(10.0, log10_hz) / 1e6


def mhz_loss_weight(pi_norm: torch.Tensor, c: Config) -> torch.Tensor:
    """Per-sample multiplier: low MHz -> ``min``, high MHz -> ``max``.

    Modes:
      - ``log``: log10(MHz) ramp (default; 10 MHz .. 500 MHz)
      - ``linear``: linear MHz ramp
      - ``piecewise``: 1.0 below threshold, ``high_mult`` at/above (legacy)
    """
    if pi_norm is None:
        raise ValueError("pi_norm required for MHz loss weights")
    if not bool(getattr(c, "heatmap_mhz_loss_weight_enabled", False)):
        return torch.ones(pi_norm.shape[0], device=pi_norm.device, dtype=torch.float32)

    mhz = pi_norm_to_mhz(pi_norm).float()
    mode = str(getattr(c, "heatmap_mhz_loss_weight_mode", "log"))
    w_min = float(getattr(c, "heatmap_mhz_loss_weight_min", 0.5))
    w_max = float(getattr(c, "heatmap_mhz_loss_weight_max", 2.5))

    if mode == "piecewise":
        thr = float(getattr(c, "layout_high_freq_mhz_threshold", 250.0))
        hi = float(getattr(c, "layout_high_freq_loss_mult", 2.0))
        return torch.where(mhz >= thr, mhz.new_tensor(hi), mhz.new_tensor(1.0))

    m_lo = float(getattr(c, "heatmap_mhz_loss_weight_min_mhz", 10.0))
    m_hi = float(getattr(c, "heatmap_mhz_loss_weight_max_mhz", 500.0))
    mhz_c = mhz.clamp(min=m_lo, max=m_hi)
    if mode == "linear":
        t = (mhz_c - m_lo) / max(m_hi - m_lo, 1e-6)
    else:
        t = (torch.log10(mhz_c.clamp(min=1e-6)) - math.log10(m_lo)) / max(
            math.log10(m_hi) - math.log10(m_lo), 1e-6
        )
    t = t.clamp(0.0, 1.0)
    return w_min + (w_max - w_min) * t


def weighted_mean(loss_per: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    w = weight.to(loss_per.dtype)
    return (loss_per * w).sum() / w.sum().clamp(min=1e-6)
