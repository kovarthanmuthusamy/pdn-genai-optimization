"""exp053 heatmap losses: Pearson + grad + log1p peak stack (hotspot / max / centroid / blob).

Phys blob runs in robust log1p space (not denormed Ω). No FG z-Huber or percentile/dynrange terms.
"""

from __future__ import annotations

import experiments.exp038_true_multi.codes.train_vae_simple as _tr
import torch
import torch.nn.functional as F

from experiments.exp053_peak_log1p_losses.codes.spatial_metrics import pearson_fg_loss
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


def _fg_percentile(flat: torch.Tensor, q: float, dim: int = 1) -> torch.Tensor:
    """Per-row quantile on flattened FG values."""
    if hasattr(torch, "quantile"):
        return torch.quantile(flat, q, dim=dim, keepdim=True)
    k = max(1, int(round(q * (flat.shape[dim] - 1))))
    return flat.kthvalue(k, dim=dim).values.unsqueeze(dim)


def hotspot_log1p_loss(
    recon_lp: torch.Tensor,
    target_lp: torch.Tensor,
    fg: torch.Tensor,
    target_z: torch.Tensor,
    bg: float,
) -> torch.Tensor:
    """Intensity-squared weighted L1 in log1p — focuses on bright peaks."""
    intens = (target_z - bg).clamp(min=0) * fg
    denom = intens.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    w = (intens / denom).pow(2)
    diff = (recon_lp - target_lp).abs()
    return (diff * w).sum((1, 2, 3)) / w.sum((1, 2, 3)).clamp(min=1e-6)


def max_log1p_loss(
    recon_lp: torch.Tensor,
    target_lp: torch.Tensor,
    fg: torch.Tensor,
) -> torch.Tensor:
    """|max log1p(recon) - max log1p(target)| on FG."""
    neg = torch.finfo(recon_lp.dtype).min
    r_fg = recon_lp.masked_fill(fg < 0.5, neg)
    t_fg = target_lp.masked_fill(fg < 0.5, neg)
    return (r_fg.amax(dim=(2, 3)) - t_fg.amax(dim=(2, 3))).abs().view(-1)


def p95_log1p_loss(
    recon_lp: torch.Tensor,
    target_lp: torch.Tensor,
    fg: torch.Tensor,
) -> torch.Tensor:
    """|p95 log1p(recon) - p95 log1p(target)| on FG pixels."""
    b = recon_lp.shape[0]
    flat_r = (recon_lp * fg).reshape(b, -1)
    flat_t = (target_lp * fg).reshape(b, -1)
    p95_r = _fg_percentile(flat_r, 0.95, dim=1).squeeze(1)
    p95_t = _fg_percentile(flat_t, 0.95, dim=1).squeeze(1)
    return (p95_r - p95_t).abs()


def centroid_top_pct_log1p_loss(
    recon_lp: torch.Tensor,
    target_lp: torch.Tensor,
    fg: torch.Tensor,
    *,
    top_q: float = 0.90,
) -> torch.Tensor:
    """L2 distance between centroids of top target-percentile FG pixels in log1p."""
    b, _, h, w = recon_lp.shape
    flat_t = (target_lp * fg).flatten(1)
    thr = _fg_percentile(flat_t, top_q, dim=1).view(b, 1, 1, 1)
    mask = (target_lp >= thr).float() * fg
    sum_m = mask.sum(dim=(2, 3)).clamp(min=1e-6)  # (B,)
    ys = torch.arange(h, device=recon_lp.device, dtype=recon_lp.dtype).view(1, 1, h, 1)
    xs = torch.arange(w, device=recon_lp.device, dtype=recon_lp.dtype).view(1, 1, 1, w)
    hn, wn = max(h - 1, 1), max(w - 1, 1)
    wr = recon_lp.clamp(min=0) * mask
    wt = target_lp * mask
    cy_r = (wr * ys).sum(dim=(2, 3)) / sum_m / hn
    cx_r = (wr * xs).sum(dim=(2, 3)) / sum_m / wn
    cy_t = (wt * ys).sum(dim=(2, 3)) / sum_m / hn
    cx_t = (wt * xs).sum(dim=(2, 3)) / sum_m / wn
    return ((cy_r - cy_t).pow(2) + (cx_r - cx_t).pow(2)).reshape(-1)


def local_max_log1p_loss(
    recon_lp: torch.Tensor,
    target_lp: torch.Tensor,
) -> torch.Tensor:
    """L1 on 3x3 max-pooled log1p maps — peak neighborhood alignment."""
    pool_r = F.max_pool2d(recon_lp, kernel_size=3, stride=1, padding=1)
    pool_t = F.max_pool2d(target_lp, kernel_size=3, stride=1, padding=1)
    return (pool_r - pool_t).abs().mean(dim=(1, 2, 3))


def _topk_mask(intens: torch.Tensor, fg: torch.Tensor, k: int) -> torch.Tensor:
    b, _, h, w = intens.shape
    flat = (intens * fg).reshape(b, -1)
    k_eff = max(1, min(int(k), flat.shape[1]))
    _, idx = flat.topk(k_eff, dim=1)
    mask = torch.zeros_like(flat)
    mask.scatter_(1, idx, 1.0)
    return mask.reshape(b, 1, h, w)


def _blob_under_over(
    recon: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    undershoot_weight: float,
    overshoot_weight: float,
) -> torch.Tensor:
    n = mask.sum((1, 2, 3)).clamp(min=1.0)
    diff_under = F.relu(target - recon).clamp(max=20.0)
    diff_over = F.relu(recon - target).clamp(max=20.0)
    under = (diff_under * mask).pow(2).sum((1, 2, 3)) / n
    over = (diff_over * mask).pow(2).sum((1, 2, 3)) / n
    return float(undershoot_weight) * under + float(overshoot_weight) * over


def heatmap_peak_log1p_bundle(
    recon: torch.Tensor,
    target: torch.Tensor,
    c: _tr.Config,
    *,
    pi_freq: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Weighted log1p peak losses — each value per-sample (B,)."""
    bg = c.background_value + 0.5
    fg = (target > bg).float()
    recon_lp, tgt_lp = _robust_log1p_maps(recon, target, c, pi_freq=pi_freq)

    out: dict[str, torch.Tensor] = {}
    w_hot = float(getattr(c, "heatmap_peak_hotspot_weight", 0.0))
    w_max = float(getattr(c, "heatmap_peak_max_log1p_weight", 0.0))
    w_p95 = float(getattr(c, "heatmap_peak_p95_log1p_weight", 0.0))
    w_ctr = float(getattr(c, "heatmap_peak_centroid_weight", 0.0))
    w_lmax = float(getattr(c, "heatmap_peak_local_max_weight", 0.0))

    if w_hot > 0:
        out["hotspot"] = torch.nan_to_num(
            hotspot_log1p_loss(recon_lp, tgt_lp, fg, target, bg), nan=0.0, posinf=1e4, neginf=0.0,
        )
    if w_max > 0:
        out["max"] = torch.nan_to_num(max_log1p_loss(recon_lp, tgt_lp, fg), nan=0.0, posinf=1e4, neginf=0.0)
    if w_p95 > 0:
        out["p95"] = torch.nan_to_num(p95_log1p_loss(recon_lp, tgt_lp, fg), nan=0.0, posinf=1e4, neginf=0.0)
    if w_ctr > 0:
        top_q = float(getattr(c, "heatmap_peak_centroid_top_q", 0.90))
        out["centroid"] = torch.nan_to_num(
            centroid_top_pct_log1p_loss(recon_lp, tgt_lp, fg, top_q=top_q),
            nan=0.0, posinf=4.0, neginf=0.0,
        )
    if w_lmax > 0:
        out["local_max"] = torch.nan_to_num(local_max_log1p_loss(recon_lp, tgt_lp), nan=0.0, posinf=1e4, neginf=0.0)

    return out


def heatmap_peak_log1p_total(
    recon: torch.Tensor,
    target: torch.Tensor,
    c: _tr.Config,
    *,
    pi_freq: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Scalar-weighted sum of log1p peak terms; returns (per_sample, raw_terms)."""
    terms = heatmap_peak_log1p_bundle(recon, target, c, pi_freq=pi_freq)
    if not terms:
        z = recon.new_zeros(recon.shape[0])
        return z, terms
    wmap = {
        "hotspot": float(getattr(c, "heatmap_peak_hotspot_weight", 0.0)),
        "max": float(getattr(c, "heatmap_peak_max_log1p_weight", 0.0)),
        "p95": float(getattr(c, "heatmap_peak_p95_log1p_weight", 0.0)),
        "centroid": float(getattr(c, "heatmap_peak_centroid_weight", 0.0)),
        "local_max": float(getattr(c, "heatmap_peak_local_max_weight", 0.0)),
    }
    total = recon.new_zeros(recon.shape[0])
    for name, term in terms.items():
        total = total + wmap.get(name, 0.0) * term.reshape(recon.shape[0])
    return torch.nan_to_num(total, nan=0.0, posinf=1e4, neginf=0.0), terms


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
    """Top-k target blob in log1p: mask = top-k(target) only (no recon union)."""
    bg = c.background_value + 0.5
    fg = (target > bg).float()
    recon, target, fg = _tr._downsample_maps_2x(recon, target, fg)

    topk_k = int(getattr(c, "heatmap_peak_topk_k", 32))
    intens_t = (target - bg).clamp(min=0)
    blob_mask = _topk_mask(intens_t, fg, topk_k)

    recon_lp, tgt_lp = _robust_log1p_maps(recon, target, c, pi_freq=pi_freq)
    under_w = float(getattr(c, "heatmap_peak_phys_undershoot_weight", 1.0))
    over_w = float(getattr(c, "heatmap_peak_phys_overshoot_weight", 2.0))
    per = _blob_under_over(recon_lp, tgt_lp, blob_mask, undershoot_weight=under_w, overshoot_weight=over_w)
    per = torch.nan_to_num(per, nan=0.0, posinf=1e4, neginf=0.0)
    return per if reduction == "none" else per.mean()
