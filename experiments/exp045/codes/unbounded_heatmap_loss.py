"""Robust unbounded heatmap loss — tail-weighted Huber, no recon clip (exp045)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import experiments.exp038_true_multi.codes.train_vae_simple as _tr
from experiments.exp045.codes.spatial_metrics import peak_loc_loss, pearson_fg_loss

_orig_huber_delta = 1.0


def _tail_weights(target: torch.Tensor, bg: float, thr: float, boost: float) -> torch.Tensor:
    """Upweight high-z foreground pixels (rare peaks)."""
    excess = (target - bg).clamp(min=0.0)
    return 1.0 + boost * torch.sigmoid((excess - thr) / max(thr * 0.25, 0.5))


def heatmap_loss_unbounded(
    recon: torch.Tensor,
    target: torch.Tensor,
    c: _tr.Config,
    ps: float,
    *,
    dynrange_weight: float | None = None,
    lite: bool = False,
) -> torch.Tensor:
    """Per-sample heatmap loss (B,) — no recon z-clip; robust Huber + tail weights."""
    return _heatmap_loss_body(
        recon, target, c, ps, dynrange_weight=dynrange_weight, lite=lite,
    )


def _heatmap_loss_body(
    recon: torch.Tensor,
    target: torch.Tensor,
    c: _tr.Config,
    ps: float,
    *,
    dynrange_weight: float | None = None,
    lite: bool = False,
) -> torch.Tensor:
    """Full heatmap loss without recon clip."""
    bg = c.background_value + 0.5
    fg, bg_m = (target > bg).float(), (target <= bg).float()
    delta = float(getattr(c, "heatmap_huber_delta", 2.0))
    huber = F.huber_loss(recon, target, delta=delta, reduction="none")
    tail_thr = float(getattr(c, "heatmap_tail_z_threshold", 3.0))
    tail_boost = float(getattr(c, "heatmap_tail_weight_boost", 2.0))
    w = _tail_weights(target, bg, tail_thr, tail_boost)
    n_fg = fg.sum((1, 2, 3)).clamp(min=1.0)
    base = (huber * fg * w).sum((1, 2, 3)) / (fg * w).sum((1, 2, 3)).clamp(min=1.0)
    intens = (target - bg).clamp(min=0)
    peak = (huber * fg * w * intens).sum((1, 2, 3)) / ((intens * fg * w).sum((1, 2, 3)).clamp(min=1e-6))
    flat_r = (recon * fg + (1 - fg) * recon.amin(dim=(2, 3), keepdim=True)).flatten(1)
    flat_t = (target * fg + (1 - fg) * target.amin(dim=(2, 3), keepdim=True)).flatten(1)
    dyn_max = F.relu(flat_t.max(1).values - flat_r.max(1).values).pow(2)
    dw = c.heatmap_dynrange_weight if dynrange_weight is None else dynrange_weight

    pearson_w = float(getattr(c, "heatmap_pearson_weight", 0.0))
    pearson = pearson_fg_loss(recon, target, fg) if pearson_w > 0 else None

    if lite:
        out = base + c.heatmap_peak_weight * peak + dw * dyn_max
        if pearson is not None:
            out = out + pearson_w * pearson
        return out

    fg_dx, fg_dy = fg[:, :, :, 1:] * fg[:, :, :, :-1], fg[:, :, 1:, :] * fg[:, :, :-1, :]
    gx = F.huber_loss(recon[:, :, :, 1:] - recon[:, :, :, :-1], target[:, :, :, 1:] - target[:, :, :, :-1],
                      delta=0.5, reduction="none")
    gy = F.huber_loss(recon[:, :, 1:, :] - recon[:, :, :-1, :], target[:, :, 1:, :] - target[:, :, :-1, :],
                      delta=0.5, reduction="none")
    grad = (gx * fg_dx).sum((1, 2, 3)) / fg_dx.sum((1, 2, 3)).clamp(min=1.0)
    grad += (gy * fg_dy).sum((1, 2, 3)) / fg_dy.sum((1, 2, 3)).clamp(min=1.0)
    lap_k = _tr._lap_k(recon.device, recon.dtype)
    stacked = torch.cat([recon, target], dim=0)
    lap_both = F.conv2d(stacked, lap_k, padding=1)
    lap_r, lap_t = lap_both.chunk(2, dim=0)
    lap = ((lap_r - lap_t).pow(2) * fg).sum((1, 2, 3)) / n_fg
    n_bg = bg_m.sum((1, 2, 3)).clamp(min=1.0)
    contrast = F.relu((recon * bg_m).sum((1, 2, 3)) / n_bg + c.heatmap_contrast_margin - (recon * fg).sum((1, 2, 3)) / n_fg)
    bg_h = (huber * bg_m).sum((1, 2, 3)) / n_bg
    p95_r = _tr._percentile_along_dim(flat_r, 0.95, dim=1)
    p95_t = _tr._percentile_along_dim(flat_t, 0.95, dim=1)
    dyn_p95 = F.relu(p95_t - p95_r).pow(2)
    dyn = dyn_max + 0.5 * dyn_p95
    peak_loc_w = float(getattr(c, "heatmap_peak_loc_weight", 0.0))
    peak_loc = peak_loc_loss(recon, target, fg, bg) if peak_loc_w > 0 else None
    out = (base + c.heatmap_peak_weight * peak + c.heatmap_grad_weight * grad
           + ps * (c.heatmap_lap_weight * lap + c.heatmap_contrast_weight * contrast
                   + c.heatmap_bg_weight * bg_h + dw * dyn))
    if pearson is not None:
        out = out + pearson_w * pearson
    if peak_loc is not None:
        out = out + peak_loc_w * peak_loc
    return out


def heatmap_phys_amplitude_loss_unbounded(
    recon: torch.Tensor,
    target: torch.Tensor,
    hm_log_mean: float,
    hm_log_std: float,
    c: _tr.Config,
) -> torch.Tensor:
    """Physical p99 loss — soft clamp at z_max instead of hard clip."""
    bg = c.background_value + 0.5
    fg = (target > bg).float()
    recon_p, target_p, fg = _tr._downsample_maps_2x(recon, target, fg)
    z_hi = float(getattr(c, "heatmap_soft_cap_z", 6.0))
    recon_z = recon_p.clamp(max=z_hi)
    recon_phys = torch.exp(recon_z * hm_log_std + hm_log_mean).clamp(min=0.0) - 1e-6
    tgt_phys = torch.exp(target_p * hm_log_std + hm_log_mean).clamp(min=0.0) - 1e-6
    fill_r = recon_phys.amin(dim=(2, 3), keepdim=True)
    fill_t = tgt_phys.amin(dim=(2, 3), keepdim=True)
    flat_r = (recon_phys * fg + (1 - fg) * fill_r).flatten(1)
    flat_t = (tgt_phys * fg + (1 - fg) * fill_t).flatten(1)
    p = float(c.heatmap_phys_p99_percentile) / 100.0
    p99_r = _tr._percentile_along_dim(flat_r, p, dim=1)
    p99_t = _tr._percentile_along_dim(flat_t, p, dim=1)
    return F.relu(p99_t - p99_r).pow(2).mean()


def patch_unbounded_losses() -> None:
    """Replace heatmap losses in exp038 trainer with unbounded variants."""
    _tr.heatmap_loss = heatmap_loss_unbounded
    _tr.heatmap_phys_amplitude_loss = heatmap_phys_amplitude_loss_unbounded

    def _no_clip_recon(recon: torch.Tensor, c: _tr.Config) -> torch.Tensor:
        if getattr(c, "heatmap_unbounded", False) or not c.heatmap_clip_recon:
            return recon
        lo, hi = c.heatmap_z_clip_min, c.heatmap_z_clip_max
        if lo is None or hi is None:
            return recon
        return recon.clamp(lo, hi)

    _tr._clip_recon_heatmap_z = _no_clip_recon
