"""Global-max heatmap normalization (physical Ω / global_max → ~[0, 1])."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def is_global_max_stats(stats: dict[str, Any]) -> bool:
    hm = stats.get("Heatmap") or {}
    return hm.get("norm_mode") == "global_max" or hm.get("global_max_ohm") is not None


def load_gmax_from_stats(stats: dict[str, Any]) -> tuple[float, float]:
    """Return (global_max_ohm, background_value)."""
    hm = stats.get("Heatmap") or {}
    gmax = float(hm["global_max_ohm"])
    bg = float(hm.get("background_value", 0.0))
    return gmax, bg


def zscore_to_physical(
    z: np.ndarray | torch.Tensor,
    *,
    log_mean: float,
    log_std: float,
) -> np.ndarray | torch.Tensor:
    """Invert log-z-score multifreq heatmaps to Ω."""
    if isinstance(z, torch.Tensor):
        return (torch.exp(z * log_std + log_mean) - 1.0).clamp(min=0.0)
    phys = np.exp(z * log_std + log_mean) - 1.0
    return np.clip(phys, 0.0, None)


def physical_to_gmax_norm(
    phys: np.ndarray,
    *,
    global_max_ohm: float,
    bg_ohm: float,
) -> np.ndarray:
    """Map physical Ω to [0, 1] with background forced to 0."""
    out = np.zeros_like(phys, dtype=np.float32)
    fg = phys > bg_ohm
    if global_max_ohm <= 0:
        raise ValueError("global_max_ohm must be positive")
    out[fg] = (phys[fg] / global_max_ohm).astype(np.float32)
    return out


def gmax_norm_to_physical(norm: torch.Tensor, global_max_ohm: float) -> torch.Tensor:
    return norm.clamp(min=0.0) * global_max_ohm


def heatmap_fg_threshold(c) -> float:
    """Norm-space FG threshold (matches ``background_value + 0.5`` convention)."""
    t = getattr(c, "heatmap_fg_threshold", None)
    if t is not None:
        return float(t)
    return float(c.background_value) + 0.5


def heatmap_phys_amplitude_loss_gmax(
    recon: torch.Tensor,
    target: torch.Tensor,
    global_max_ohm: float,
    c,
    *,
    downsample_2x=None,
    percentile_fn=None,
) -> torch.Tensor:
    """FG p99 in Ω — for global-max normalized heatmaps."""
    from experiments.exp038_true_multi.codes import train_vae_simple as _tr

    clip = _tr._clip_recon_heatmap_z(recon, c)
    bg = heatmap_fg_threshold(c)
    fg = (target > bg).float()
    if downsample_2x is None:
        downsample_2x = _tr._downsample_maps_2x
    if percentile_fn is None:
        percentile_fn = _tr._percentile_along_dim
    recon_d, target_d, fg_d = downsample_2x(clip, target, fg)
    recon_p = gmax_norm_to_physical(recon_d, global_max_ohm)
    tgt_p = gmax_norm_to_physical(target_d, global_max_ohm)
    fill_r = recon_p.amin(dim=(2, 3), keepdim=True)
    fill_t = tgt_p.amin(dim=(2, 3), keepdim=True)
    flat_r = (recon_p * fg_d + (1.0 - fg_d) * fill_r).flatten(1)
    flat_t = (tgt_p * fg_d + (1.0 - fg_d) * fill_t).flatten(1)
    q = float(c.heatmap_phys_p99_percentile) / 100.0
    p99_r = percentile_fn(flat_r, q, dim=1, max_samples=1024)
    p99_t = percentile_fn(flat_t, q, dim=1, max_samples=1024)
    return torch.relu(p99_t - p99_r).pow(2).mean()


def load_stats(data_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(data_dir) / "normalization_stats.json").read_text(encoding="utf-8"))
