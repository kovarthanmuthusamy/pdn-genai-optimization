"""Log-z heatmap clipping and denormalization helpers.

Delegates scale/denorm to ``norm_stats`` (single source of truth from normalization_stats.json).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src_vae.others.norm_stats import (
    HeatmapNormStats,
    load_heatmap_clip_bounds as _load_clip_from_norm_stats,
    load_heatmap_stats,
)


def load_heatmap_z_clip_bounds(
    data_dir: str | Path,
    *,
    stats_path: str | Path | None = None,
) -> tuple[float, float] | None:
    return _load_clip_from_norm_stats(data_dir, stats_path=stats_path)


def clip_heatmap_z(
    x: torch.Tensor | np.ndarray,
    lo: float,
    hi: float,
) -> torch.Tensor | np.ndarray:
    if isinstance(x, np.ndarray):
        return np.clip(x, lo, hi)
    return x.clamp(lo, hi)


def clip_fraction(x: torch.Tensor | np.ndarray, lo: float, hi: float) -> float:
    if isinstance(x, torch.Tensor):
        a = x.detach().float()
        n = a.numel()
        if n == 0:
            return 0.0
        hit = ((a < lo) | (a > hi)).sum().item()
        return float(hit) / float(n)
    a = np.asarray(x, dtype=np.float64)
    n = a.size
    if n == 0:
        return 0.0
    return float(((a < lo) | (a > hi)).sum()) / float(n)


def heatmap_z_to_physical(
    hm_z: torch.Tensor,
    log_mean: float,
    log_std: float,
    *,
    clip_lo: float | None = None,
    clip_hi: float | None = None,
    hm_stats: HeatmapNormStats | dict[str, Any] | None = None,
    mhz: float | torch.Tensor | None = None,
    pi_norm: torch.Tensor | None = None,
) -> torch.Tensor:
    """Denorm heatmap to physical Ω.

    Prefer ``hm_stats`` (or pass ``HeatmapNormStats`` via dict with norm_mode).
    Legacy ``log_mean``/``log_std`` used when ``hm_stats`` is None.
    """
    if hm_stats is not None:
        if isinstance(hm_stats, dict):
            stats = HeatmapNormStats.from_json({"Heatmap": hm_stats, "background_value": hm_stats.get("background_value", -3.0)})
        else:
            stats = hm_stats
        apply = clip_lo is not None and clip_hi is not None
        if not apply:
            apply = True
        return stats.norm_to_physical(
            hm_z, mhz=mhz, pi_norm=pi_norm, apply_clip=apply,
        )

    z = hm_z
    if clip_lo is not None and clip_hi is not None:
        z = z.clamp(clip_lo, clip_hi)
    return (torch.exp(z * log_std + log_mean) - 1.0).clamp(min=0.0)


def heatmap_norm_to_physical(
    hm_norm: torch.Tensor,
    hm_stats: dict[str, Any] | HeatmapNormStats,
    *,
    clip_lo: float | None = None,
    clip_hi: float | None = None,
    mhz: float | torch.Tensor | None = None,
    pi_norm: torch.Tensor | None = None,
) -> torch.Tensor:
    """Denorm heatmap tensor to Ω (z-score, robust per-MHz, or global-max)."""
    if isinstance(hm_stats, HeatmapNormStats):
        stats = hm_stats
    else:
        stats = HeatmapNormStats.from_json({"Heatmap": hm_stats})
    apply = clip_lo is not None and clip_hi is not None
    if not apply and stats.is_robust_per_mhz() and mhz is not None:
        b = stats.bin_stats(float(mhz) if not isinstance(mhz, torch.Tensor) else float(mhz.item()))
        clip_lo, clip_hi = b.clip_min, b.clip_max
        apply = True
    return stats.norm_to_physical(
        hm_norm, mhz=mhz, pi_norm=pi_norm, apply_clip=apply,
    )


def describe_clip_bounds(data_dir: str | Path) -> str:
    try:
        hm = load_heatmap_stats(data_dir)
        return f"heatmap norm: {hm.describe()}"
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return "heatmap clip: no normalization_stats.json"
