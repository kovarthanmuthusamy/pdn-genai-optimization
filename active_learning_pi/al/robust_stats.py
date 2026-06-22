"""Robust foreground peak statistics on physical heatmaps.

Run:
    Import only — used by ``inference_pool`` and ``evaluate_off_anchor``."""
from __future__ import annotations

import numpy as np


def foreground_mask(hm: np.ndarray, bg_margin: float = 0.5) -> np.ndarray:
    """hm: (2,H,W) z-score or (H,W) single channel — use channel 0."""
    if hm.ndim == 3:
        ch = hm[0]
    else:
        ch = hm
    return ch > bg_margin


def robust_peak_stats(hm_phys: np.ndarray, mask_board: np.ndarray | None = None) -> dict[str, float]:
    """
    hm_phys: (2,H,W) physical impedance × mask or (H,W).
    Returns foreground p95, p99, top-k mean (k=1% of fg pixels).
    """
    if hm_phys.ndim == 3:
        z = hm_phys[0] * hm_phys[1] if hm_phys.shape[0] >= 2 else hm_phys[0]
    else:
        z = hm_phys
    fg = z > 1e-6
    if mask_board is not None:
        fg = fg & (mask_board > 0.5)
    vals = z[fg]
    if vals.size == 0:
        return {"p95": 0.0, "p99": 0.0, "topk_mean": 0.0, "fg_pixels": 0}
    k = max(1, int(0.01 * vals.size))
    topk = np.partition(vals, -k)[-k:]
    return {
        "p95": float(np.percentile(vals, 95)),
        "p99": float(np.percentile(vals, 99)),
        "topk_mean": float(topk.mean()),
        "fg_pixels": int(vals.size),
    }


def metric_from_stats(stats_list: list[dict[str, float]], key: str = "p99") -> float:
    arr = np.array([s[key] for s in stats_list], dtype=np.float64)
    return float(arr.var()) if arr.size > 1 else 0.0
