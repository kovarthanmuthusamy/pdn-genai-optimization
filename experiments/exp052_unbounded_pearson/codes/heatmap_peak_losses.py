"""exp052 heatmap losses: FG Pearson (global) + gradient field (local) + log1p peak blob.

Phys blob runs in robust log1p space (not denormed Ω) so unbounded z cannot explode via exp().
No percentile / dynrange / Huber FG MSE terms.
"""

from __future__ import annotations

import experiments.exp038_true_multi.codes.train_vae_simple as _tr
import torch
import torch.nn.functional as F

from experiments.exp052_unbounded_pearson.codes.spatial_metrics import pearson_fg_loss
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
    """FG-masked grad loss: vector huber + flow direction."""
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
    out = w_vec * vec_loss + w_dir * dir_loss
    return torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=0.0)


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


def _robust_log1p_maps(
    recon: torch.Tensor,
    target: torch.Tensor,
    c: _tr.Config,
    *,
    pi_freq: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map z-score heatmaps to log1p(Ω) per MHz bin (linear; no exp blow-up)."""
    bundle: NormStatsBundle | None = getattr(c, "_norm_stats", None)
    if bundle is not None and bundle.heatmap.is_robust_per_mhz() and pi_freq is not None:
        hm = bundle.heatmap
        rows_r, rows_t = [], []
        for i in range(recon.shape[0]):
            b = hm.bin_stats(pi_norm=pi_freq[i : i + 1])
            rows_r.append(recon[i : i + 1] * b.iqr + b.median)
            rows_t.append(target[i : i + 1] * b.iqr + b.median)
        return torch.cat(rows_r, dim=0), torch.cat(rows_t, dim=0)

    if bundle is not None:
        b = bundle.heatmap.bin_stats()
        return recon * b.log_std + b.log_mean, target * b.log_std + b.log_mean

    hm_log_mean = float(getattr(c, "_hm_log_mean", 0.0))
    hm_log_std = float(getattr(c, "_hm_log_std", 1.0))
    return recon * hm_log_std + hm_log_mean, target * hm_log_std + hm_log_mean


def _blob_under_over(
    recon: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    overshoot_weight: float,
) -> torch.Tensor:
    n = mask.sum((1, 2, 3)).clamp(min=1.0)
    diff_under = F.relu(target - recon).clamp(max=20.0)
    diff_over = F.relu(recon - target).clamp(max=20.0)
    under = (diff_under * mask).pow(2).sum((1, 2, 3)) / n
    over = (diff_over * mask).pow(2).sum((1, 2, 3)) / n
    return under + float(overshoot_weight) * over


def heatmap_loss_pearson_grad(
    recon: torch.Tensor,
    target: torch.Tensor,
    c: _tr.Config,
    *,
    lite: bool = False,
) -> torch.Tensor:
    """FG Pearson (global) + grad field (local) — per-sample (B,)."""
    bg = c.background_value + 0.5
    fg = (target > bg).float()
    pearson_w = float(getattr(c, "heatmap_pearson_weight", 2.5))
    pearson = pearson_fg_loss(recon, target, fg) * pearson_w
    if lite:
        return torch.nan_to_num(pearson, nan=2.0, posinf=2.0, neginf=0.0)
    out = pearson + grad_vector_field_loss(recon, target, fg, c)
    return torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=0.0)


heatmap_loss_tier_a = heatmap_loss_pearson_grad


def heatmap_peak_blob_phys_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    c: _tr.Config,
    *,
    pi_freq: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Top-k peak blob in robust log1p space: under-predict + overshoot."""
    bg = c.background_value + 0.5
    fg = (target > bg).float()
    recon, target, fg = _tr._downsample_maps_2x(recon, target, fg)

    topk_k = int(getattr(c, "heatmap_peak_topk_k", 32))
    blob_mask = _topk_target_mask(target, fg, bg, topk_k)
    recon_lp, tgt_lp = _robust_log1p_maps(recon, target, c, pi_freq=pi_freq)
    over_w = float(getattr(c, "heatmap_peak_phys_overshoot_weight", 2.0))
    per = _blob_under_over(recon_lp, tgt_lp, blob_mask, overshoot_weight=over_w)
    per = torch.nan_to_num(per, nan=0.0, posinf=1e4, neginf=0.0)
    return per if reduction == "none" else per.mean()
