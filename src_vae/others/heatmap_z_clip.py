"""Clip log-z heatmaps to dataset tail bounds (minimal data loss).

Uses ``Heatmap.clip_min`` / ``clip_max`` from ``normalization_stats.json``
(typically ~1st–99th percentile of foreground z, not the full z_min/z_max).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_heatmap_z_clip_bounds(
    data_dir: str | Path,
    *,
    stats_path: str | Path | None = None,
) -> tuple[float, float] | None:
    """Return (clip_min, clip_max) or None if stats missing."""
    if stats_path is None:
        stats_path = Path(data_dir) / "normalization_stats.json"
    p = Path(stats_path)
    if not p.is_file():
        return None
    raw = json.loads(p.read_text(encoding="utf-8"))
    hm = raw.get("Heatmap") or {}
    lo = hm.get("clip_min")
    hi = hm.get("clip_max")
    if lo is None or hi is None:
        return None
    return float(lo), float(hi)


def clip_heatmap_z(
    x: torch.Tensor | np.ndarray,
    lo: float,
    hi: float,
) -> torch.Tensor | np.ndarray:
    if isinstance(x, np.ndarray):
        return np.clip(x, lo, hi)
    return x.clamp(lo, hi)


def clip_fraction(x: torch.Tensor | np.ndarray, lo: float, hi: float) -> float:
    """Fraction of elements that would change under clip (for logging)."""
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
) -> torch.Tensor:
    """Denorm log-z heatmap to Ω with optional z clip before ``exp``."""
    z = hm_z
    if clip_lo is not None and clip_hi is not None:
        z = z.clamp(clip_lo, clip_hi)
    return (torch.exp(z * log_std + log_mean) - 1.0).clamp(min=0.0)


def heatmap_norm_to_physical(
    hm_norm: torch.Tensor,
    hm_stats: dict[str, Any],
    *,
    clip_lo: float | None = None,
    clip_hi: float | None = None,
) -> torch.Tensor:
    """Denorm heatmap tensor to Ω (z-score or global-max)."""
    if hm_stats.get("norm_mode") == "global_max" or hm_stats.get("global_max_ohm") is not None:
        gmax = float(hm_stats["global_max_ohm"])
        z = hm_norm
        if clip_lo is not None and clip_hi is not None:
            z = z.clamp(clip_lo, clip_hi)
        return z.clamp(min=0.0) * gmax
    log_mean = float(hm_stats["log_mean"])
    log_std = float(hm_stats["log_std"])
    return heatmap_z_to_physical(
        hm_norm, log_mean, log_std, clip_lo=clip_lo, clip_hi=clip_hi,
    )


def describe_clip_bounds(data_dir: str | Path) -> str:
    """Human-readable summary vs full z range in stats."""
    p = Path(data_dir) / "normalization_stats.json"
    if not p.is_file():
        return "heatmap clip: no normalization_stats.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    hm: dict[str, Any] = raw.get("Heatmap") or {}
    lo, hi = hm.get("clip_min"), hm.get("clip_max")
    if lo is None or hi is None:
        return "heatmap clip: clip_min/max missing in stats"
    if hm.get("norm_mode") == "global_max" or hm.get("global_max_ohm") is not None:
        gmax = float(hm.get("global_max_ohm", 0.0))
        return (
            f"heatmap global-max: norm∈[{lo:.4f}, {hi:.4f}]  "
            f"global_max_ohm={gmax:.4f}  bg={hm.get('background_value', 0.0)}"
        )
    zmin, zmax = hm.get("z_min"), hm.get("z_max")
    parts = [f"clip z∈[{lo:.4f}, {hi:.4f}]"]
    if zmin is not None and zmax is not None:
        parts.append(f"full data z∈[{zmin:.4f}, {zmax:.4f}]")
    return "heatmap z-clip: " + "  ".join(parts)
