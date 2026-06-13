"""PI frequency helpers for inference on unseen / off-anchor MHz (between training anchors)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from experiments.exp038_true_multi.codes.dataloader_multifreq import ANCHOR_MHZ
from src_vae.others.pi_freq_utils import pi_freq_mhz_to_norm

# Typical foreground max (physical Ω) per training anchor — from multifreq dataset stats.
# Used to calibrate generated maps at off-anchor MHz via log-interpolation.
DEFAULT_ANCHOR_FG_MAX: dict[float, float] = {
    10.0: 2.5,
    63.0: 5.5,
    130.0: 9.0,
    200.0: 14.0,
    270.0: 18.0,
    400.0: 22.0,
    500.0: 26.0,
}


def bracket_anchors_mhz(mhz: float, anchors: Sequence[float] = ANCHOR_MHZ) -> tuple[float, float, float]:
    """Return (mhz_lo, mhz_hi, t) with t in [0,1] for log-spaced blend weight on mhz_hi."""
    mhz = float(mhz)
    a = sorted(float(x) for x in anchors)
    if mhz <= a[0]:
        return a[0], a[0], 0.0
    if mhz >= a[-1]:
        return a[-1], a[-1], 1.0
    for i in range(len(a) - 1):
        if a[i] <= mhz <= a[i + 1]:
            lo, hi = a[i], a[i + 1]
            if hi <= lo:
                return lo, hi, 0.0
            t = (math.log10(mhz) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
            return lo, hi, float(np.clip(t, 0.0, 1.0))
    return a[-1], a[-1], 1.0


def is_training_anchor(mhz: float, tol: float = 0.5) -> bool:
    return any(abs(float(mhz) - float(a)) <= tol for a in ANCHOR_MHZ)


def interp_anchor_value(mhz: float, table: dict[float, float]) -> float:
    lo, hi, t = bracket_anchors_mhz(mhz)
    v_lo = float(table.get(lo, table.get(float(lo), 0.0)))
    v_hi = float(table.get(hi, table.get(float(hi), v_lo)))
    if lo == hi:
        return v_lo
    return (1.0 - t) * v_lo + t * v_hi


def load_anchor_fg_max_table(path: Path | None = None) -> dict[float, float]:
    if path is None:
        path = Path(__file__).resolve().parents[1] / "metrics" / "anchor_hm_fg_max.json"
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {float(k): float(v) for k, v in raw.items()}
    return dict(DEFAULT_ANCHOR_FG_MAX)


def _heatmap_plane(hm: np.ndarray) -> np.ndarray:
    """Return (H, W) for masking — accepts (H, W) or (1, H, W) / (C, H, W)."""
    a = np.asarray(hm)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        return a[0] if a.shape[0] != 1 else np.squeeze(a, axis=0)
    raise ValueError(f"Expected 2D or 3D heatmap, got shape {a.shape}")


def calibrate_heatmap_physical(
    hm_phys: np.ndarray,
    mask: np.ndarray,
    target_fg_max: float,
    *,
    also_p95: float | None = None,
    p95_blend: float = 0.35,
) -> np.ndarray:
    """Scale physical heatmap so foreground max (and optionally p95) match targets."""
    out = np.asarray(hm_phys, dtype=np.float32).copy()
    plane = _heatmap_plane(out)
    m = np.asarray(mask, dtype=bool)
    if m.shape != plane.shape:
        raise ValueError(f"mask shape {m.shape} != heatmap plane {plane.shape}")
    fg = plane[m]
    if fg.size == 0 or target_fg_max <= 0:
        return out
    cur_max = float(fg.max())
    if cur_max < 1e-8:
        return out
    scale = float(target_fg_max) / cur_max
    if also_p95 is not None and also_p95 > 0:
        cur_p95 = float(np.percentile(fg, 95))
        if cur_p95 > 1e-8:
            scale_p95 = float(also_p95) / cur_p95
            scale = (1.0 - p95_blend) * scale + p95_blend * scale_p95
    out = (out * scale).clip(min=0.0)
    return out


def pi_norm_tensor(mhz: float, batch: int, device: torch.device) -> torch.Tensor:
    v = pi_freq_mhz_to_norm(mhz)
    return torch.full((batch,), v, dtype=torch.float32, device=device)
