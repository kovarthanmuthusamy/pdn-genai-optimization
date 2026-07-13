"""Robust per-MHz heatmap normalization for active-learning labels.

Matches ``pipelines/normalize/multifreq.py`` rules used by
``datasets/data_multifreq_train_norm_unbounded``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


def _ensure_repo_on_path(groot: Path) -> None:
    if str(groot) not in sys.path:
        sys.path.insert(0, str(groot))


def load_heatmap_stats(stats_json: Path) -> dict[str, Any]:
    import json

    raw = json.loads(stats_json.read_text(encoding="utf-8"))
    return dict(raw["Heatmap"])


def is_robust_per_mhz_stats(heatmap_stats: dict[str, Any]) -> bool:
    mode = str(heatmap_stats.get("norm_mode", ""))
    return mode in ("robust_log1p_per_mhz", "robust_log1p_per_mhz_unbounded") or bool(
        heatmap_stats.get("by_mhz")
    )


def normalize_heatmap_raw(
    raw: np.ndarray,
    hm_stats: dict[str, Any],
    *,
    mhz: float | None = None,
    groot: Path | None = None,
) -> np.ndarray:
    """Normalize raw (2,H,W) or (1,H,W) heatmap using training stats."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim == 3 and raw.shape[0] == 1:
        return raw.astype(np.float32, copy=False)

    if is_robust_per_mhz_stats(hm_stats):
        if mhz is None:
            raise ValueError("mhz is required for robust per-MHz heatmap normalization")
        if groot is not None:
            _ensure_repo_on_path(groot)
        from pipelines.normalize.multifreq import _normalize_raw_heatmap_to_array

        return _normalize_raw_heatmap_to_array(raw, hm_stats, mhz=float(mhz))

    if groot is not None:
        _ensure_repo_on_path(groot)
    from pipelines.normalize.multifreq import _normalize_raw_heatmap_to_array

    return _normalize_raw_heatmap_to_array(raw, hm_stats, mhz=mhz)
