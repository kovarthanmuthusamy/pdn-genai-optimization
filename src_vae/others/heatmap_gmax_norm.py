"""Global-max heatmap normalization utilities.

Run: Imported by training scripts and ``datasets/build_multifreq_gmax_dataset.py``."""
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


LOG1P_TRAIN_SPACE = "log1p_gmax"
LINEAR_TRAIN_SPACE = "linear"


def log1p_gmax_denom(global_max_ohm: float) -> float:
    return float(np.log1p(global_max_ohm))


def linear_norm_to_log1p_train(
    norm: np.ndarray | torch.Tensor,
    global_max_ohm: float,
) -> np.ndarray | torch.Tensor:
    """Map on-disk linear norm (phys/gmax) → log1p train space in [0, 1]."""
    denom = log1p_gmax_denom(global_max_ohm)
    if isinstance(norm, torch.Tensor):
        n = norm.clamp(min=0.0)
        return torch.log1p(n * global_max_ohm) / denom
    n = np.clip(norm, 0.0, None)
    return (np.log1p(n * global_max_ohm) / denom).astype(np.float32)


def log1p_train_to_linear_norm(
    train: np.ndarray | torch.Tensor,
    global_max_ohm: float,
) -> np.ndarray | torch.Tensor:
    """Inverse: log1p train space → on-disk linear norm."""
    denom = log1p_gmax_denom(global_max_ohm)
    if isinstance(train, torch.Tensor):
        t = train.clamp(min=0.0)
        return torch.expm1(t * denom) / global_max_ohm
    t = np.clip(train, 0.0, None)
    return (np.expm1(t * denom) / global_max_ohm).astype(np.float32)


def log1p_train_to_physical(train: torch.Tensor, global_max_ohm: float) -> torch.Tensor:
    """log1p train space → Ω (skips linear norm intermediate)."""
    denom = log1p_gmax_denom(global_max_ohm)
    return torch.expm1(train.clamp(min=0.0) * denom)


def linear_threshold_to_train(thr_linear: float, global_max_ohm: float) -> float:
    if thr_linear <= 0.0:
        return 0.0
    return float(linear_norm_to_log1p_train(np.array([thr_linear], dtype=np.float32), global_max_ohm)[0])


def linear_clip_bounds_to_train(
    clip_lo: float,
    clip_hi: float,
    global_max_ohm: float,
) -> tuple[float, float]:
    return (
        linear_threshold_to_train(clip_lo, global_max_ohm) if clip_lo > 0 else 0.0,
        linear_threshold_to_train(clip_hi, global_max_ohm),
    )


def is_log1p_train_space(c) -> bool:
    return getattr(c, "heatmap_train_space", LINEAR_TRAIN_SPACE) == LOG1P_TRAIN_SPACE


def disk_to_train_space(hm: torch.Tensor, c) -> torch.Tensor:
    """On-disk linear gmax norm → model train space."""
    if not is_log1p_train_space(c):
        return hm
    return linear_norm_to_log1p_train(hm, float(c.global_max_ohm))


def train_to_disk_space(hm: torch.Tensor, c) -> torch.Tensor:
    """Model train space → on-disk linear gmax norm."""
    if not is_log1p_train_space(c):
        return hm
    return log1p_train_to_linear_norm(hm, float(c.global_max_ohm))


def heatmap_model_to_physical(hm: torch.Tensor, c) -> torch.Tensor:
    """Model output in train space → physical Ω."""
    gmax = float(c.global_max_ohm)
    if is_log1p_train_space(c):
        return log1p_train_to_physical(hm, gmax)
    return gmax_norm_to_physical(hm, gmax)


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
    if is_log1p_train_space(c):
        recon_p = log1p_train_to_physical(recon_d, global_max_ohm)
        tgt_p = log1p_train_to_physical(target_d, global_max_ohm)
    else:
        recon_p = gmax_norm_to_physical(recon_d, global_max_ohm)
        tgt_p = gmax_norm_to_physical(target_d, global_max_ohm)
    fill_r = recon_p.amin(dim=(2, 3), keepdim=True)
    fill_t = tgt_p.amin(dim=(2, 3), keepdim=True)
    flat_r = (recon_p * fg_d + (1.0 - fg_d) * fill_r).flatten(1)
    flat_t = (tgt_p * fg_d + (1.0 - fg_d) * fill_t).flatten(1)
    q = float(c.heatmap_phys_p99_percentile) / 100.0
    p99_r = percentile_fn(flat_r, q, dim=1, max_samples=1024)
    p99_t = percentile_fn(flat_t, q, dim=1, max_samples=1024)
    under = torch.relu(p99_t - p99_r).pow(2)
    over_w = float(getattr(c, "heatmap_phys_p99_over_weight", 0.5))
    over = torch.relu(p99_r - p99_t).pow(2)
    return (under + over_w * over).mean()


def load_stats(data_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(data_dir) / "normalization_stats.json").read_text(encoding="utf-8"))
