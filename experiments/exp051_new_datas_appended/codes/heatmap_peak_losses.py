"""Tier A heatmap losses for exp051.

FG huber + FG-masked gradient vector/direction + physical top-k blob (Ω).
Background contributes zero to gradient terms. No percentile training.
"""

from __future__ import annotations

import experiments.exp038_true_multi.codes.train_vae_simple as _tr
import torch
import torch.nn.functional as F

from src_vae.others.norm_stats import NormStatsBundle

_SOBEL_CACHE: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}


def _sobel_kernels(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    key = (str(device), str(dtype))
    if key not in _SOBEL_CACHE:
        gx = torch.tensor([[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]], dtype=dtype, device=device)
        gy = torch.tensor([[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]], dtype=dtype, device=device)
        _SOBEL_CACHE[key] = (gx, gy)
    return _SOBEL_CACHE[key]


def _sobel_grad(hm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gx_k, gy_k = _sobel_kernels(hm.device, hm.dtype)
    return F.conv2d(hm, gx_k, padding=1), F.conv2d(hm, gy_k, padding=1)


def _masked_mean(per_px: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    n = mask.sum((1, 2, 3)).clamp(min=1.0)
    return (per_px * mask).sum((1, 2, 3)) / n


def grad_vector_field_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    fg: torch.Tensor,
    c: _tr.Config,
) -> torch.Tensor:
    """FG-masked ∇ loss: vector huber + flow direction (background = 0)."""
    gr_x, gr_y = _sobel_grad(recon)
    gt_x, gt_y = _sobel_grad(target)

    min_mag = float(getattr(c, "heatmap_grad_direction_min_mag", 0.08))
    huber_delta = float(getattr(c, "heatmap_grad_huber_delta", 0.5))
    eps = 1e-6

    mag_t = torch.sqrt(gt_x.pow(2) + gt_y.pow(2) + eps)
    active_flow = fg * (mag_t > min_mag).float()

    vec_x = F.huber_loss(gr_x, gt_x, delta=huber_delta, reduction="none")
    vec_y = F.huber_loss(gr_y, gt_y, delta=huber_delta, reduction="none")
    vec_loss = _masked_mean(vec_x + vec_y, fg)

    mag_r = torch.sqrt(gr_x.pow(2) + gr_y.pow(2) + eps)
    cos = (gr_x * gt_x + gr_y * gt_y) / (mag_r * mag_t)
    dir_loss = _masked_mean(1.0 - cos.clamp(-1.0, 1.0), active_flow)

    w_vec = float(getattr(c, "heatmap_grad_vector_weight", 2.0))
    w_dir = float(getattr(c, "heatmap_grad_direction_weight", 1.0))
    return w_vec * vec_loss + w_dir * dir_loss


def _topk_target_mask(
    target: torch.Tensor,
    fg: torch.Tensor,
    bg: float,
    k: int,
) -> torch.Tensor:
    intens = (target - bg).clamp(min=0.0) * fg
    b, _, h, w = intens.shape
    flat = intens.reshape(b, -1)
    k_eff = max(1, min(int(k), flat.shape[1]))
    _, idx = flat.topk(k_eff, dim=1)
    mask = torch.zeros_like(flat)
    mask.scatter_(1, idx, 1.0)
    return mask.reshape(b, 1, h, w)


def _norm_to_physical_maps(
    recon: torch.Tensor,
    target: torch.Tensor,
    c: _tr.Config,
    *,
    pi_freq: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    bundle: NormStatsBundle | None = getattr(c, "_norm_stats", None)
    apply_clip = not (bundle is not None and bundle.heatmap.is_unbounded())
    if bundle is not None:
        hm = bundle.heatmap
        if hm.is_robust_per_mhz() and pi_freq is not None:
            recon_p = hm.norm_to_physical(recon, pi_norm=pi_freq, apply_clip=apply_clip) - 1e-6
            tgt_p = hm.norm_to_physical(target, pi_norm=pi_freq, apply_clip=apply_clip) - 1e-6
        else:
            b = hm.bin_stats()
            recon_p = torch.exp(recon * b.log_std + b.log_mean).clamp(min=0.0) - 1e-6
            tgt_p = torch.exp(target * b.log_std + b.log_mean).clamp(min=0.0) - 1e-6
    else:
        hm_log_mean = float(getattr(c, "_hm_log_mean", 0.0))
        hm_log_std = float(getattr(c, "_hm_log_std", 1.0))
        recon_p = torch.exp(recon * hm_log_std + hm_log_mean).clamp(min=0.0) - 1e-6
        tgt_p = torch.exp(target * hm_log_std + hm_log_mean).clamp(min=0.0) - 1e-6
    return recon_p, tgt_p


def _blob_under_over(
    recon: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    overshoot_weight: float,
) -> torch.Tensor:
    n = mask.sum((1, 2, 3)).clamp(min=1.0)
    under = (F.relu(target - recon) * mask).pow(2).sum((1, 2, 3)) / n
    over = (F.relu(recon - target) * mask).pow(2).sum((1, 2, 3)) / n
    return under + float(overshoot_weight) * over


def heatmap_loss_tier_a(
    recon: torch.Tensor,
    target: torch.Tensor,
    c: _tr.Config,
) -> torch.Tensor:
    """FG huber + grad vector/direction — per-sample (B,)."""
    recon = _tr._clip_recon_heatmap_z(recon, c)
    bg = c.background_value + 0.5
    fg = (target > bg).float()
    huber = F.huber_loss(recon, target, delta=1.0, reduction="none")
    n_fg = fg.sum((1, 2, 3)).clamp(min=1.0)
    base = (huber * fg).sum((1, 2, 3)) / n_fg
    return base + grad_vector_field_loss(recon, target, fg, c)


def heatmap_peak_blob_phys_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    c: _tr.Config,
    *,
    pi_freq: torch.Tensor | None = None,
) -> torch.Tensor:
    """Physical Ω top-k blob: under-predict + overshoot (scalar mean)."""
    recon = _tr._clip_recon_heatmap_z(recon, c)
    bg = c.background_value + 0.5
    fg = (target > bg).float()
    recon, target, fg = _tr._downsample_maps_2x(recon, target, fg)

    recon_p, tgt_p = _norm_to_physical_maps(recon, target, c, pi_freq=pi_freq)
    topk_k = int(getattr(c, "heatmap_peak_topk_k", 24))
    blob_mask = _topk_target_mask(tgt_p, fg, 0.0, topk_k)
    over_w = float(getattr(c, "heatmap_peak_phys_overshoot_weight", 1.75))
    return _blob_under_over(recon_p, tgt_p, blob_mask, overshoot_weight=over_w).mean()
