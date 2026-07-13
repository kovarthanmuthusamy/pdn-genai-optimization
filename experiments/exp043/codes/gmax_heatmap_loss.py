"""Global-max heatmap losses for exp043 (linear [0,1] norm space).

Replaces log-z-scaled huber deltas and adds symmetric dynrange penalties.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src_vae.others.heatmap_gmax_norm import heatmap_fg_threshold


def _fg_bg_masks(target: torch.Tensor, c) -> tuple[torch.Tensor, torch.Tensor, float]:
    bg = heatmap_fg_threshold(c)
    fg = (target > bg).float()
    bg_m = (target <= bg).float()
    return fg, bg_m, bg


def _fg_minmax_norm(flat: torch.Tensor, fg: torch.Tensor) -> torch.Tensor:
    """Per-sample min-max on foreground pixels (B, N)."""
    big, small = 1e6, -1e6
    masked_hi = torch.where(fg > 0.5, flat, torch.full_like(flat, small))
    masked_lo = torch.where(fg > 0.5, flat, torch.full_like(flat, big))
    lo = masked_lo.min(dim=1, keepdim=True).values
    hi = masked_hi.max(dim=1, keepdim=True).values
    out = (flat - lo) / (hi - lo + 1e-6)
    return out * fg


def _fg_pattern_correlation_loss(
    flat_r: torch.Tensor,
    flat_t: torch.Tensor,
    fg: torch.Tensor,
) -> torch.Tensor:
    """(1 - Pearson r)² on min-max normalized FG maps — targets spatial morphology."""
    r_n = _fg_minmax_norm(flat_r, fg)
    t_n = _fg_minmax_norm(flat_t, fg)
    r_c = r_n - (r_n * fg).sum(dim=1, keepdim=True) / fg.sum(dim=1, keepdim=True).clamp(min=1.0)
    t_c = t_n - (t_n * fg).sum(dim=1, keepdim=True) / fg.sum(dim=1, keepdim=True).clamp(min=1.0)
    corr = F.cosine_similarity(r_c, t_c, dim=1).clamp(-1.0, 1.0)
    return (1.0 - corr).pow(2)


def _fg_spread_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    fg: torch.Tensor,
    flat_r: torch.Tensor,
    flat_t: torch.Tensor,
    percentile_fn,
) -> torch.Tensor:
    """Penalize sparse fields and magnitude overshoot (edge spikes / runaway max)."""
    n_fg = fg.sum((1, 2, 3)).clamp(min=1.0)
    mean_r = (recon * fg).sum((1, 2, 3)) / n_fg
    mean_t = (target * fg).sum((1, 2, 3)) / n_fg
    mean_gap = F.relu(mean_t - mean_r).pow(2) + F.relu(mean_r - mean_t).pow(2)
    p95_r = percentile_fn(flat_r, 0.95, dim=1)
    p95_t = percentile_fn(flat_t, 0.95, dim=1)
    p95_gap = F.relu(p95_t * 0.7 - p95_r).pow(2) + 1.5 * F.relu(p95_r - p95_t).pow(2)
    return mean_gap + p95_gap


def _fg_hotspot_loss(
    huber: torch.Tensor,
    target: torch.Tensor,
    fg: torch.Tensor,
    bg: float,
) -> torch.Tensor:
    """Intensity-squared weighted huber — focuses on localized high-Ω peaks."""
    intens = ((target - bg).clamp(min=0) * fg)
    denom = intens.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    w = (intens / denom).pow(2)
    return (huber * w).sum((1, 2, 3)) / w.sum((1, 2, 3)).clamp(min=1e-6)


def _fg_peak_centroid_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    fg: torch.Tensor,
    percentile_fn,
    *,
    top_q: float = 0.92,
) -> torch.Tensor:
    """L2 distance between centroids of top target-percentile FG pixels."""
    b, _, h, w = recon.shape
    flat_t = (target * fg).flatten(1)
    thr = percentile_fn(flat_t, top_q, dim=1).view(b, 1, 1, 1)
    mask = (target >= thr).float() * fg
    sum_m = mask.sum((2, 3)).clamp(min=1e-6)
    ys = torch.arange(h, device=recon.device, dtype=recon.dtype).view(1, 1, h, 1)
    xs = torch.arange(w, device=recon.device, dtype=recon.dtype).view(1, 1, 1, w)
    wr = recon.clamp(min=0) * mask
    wt = target * mask
    # Normalized grid coords in [0, 1] — raw pixel indices (0..63) made this term O(10³–10⁴).
    hn = max(h - 1, 1)
    wn = max(w - 1, 1)
    cy_r = (wr * ys).sum((2, 3)) / sum_m / hn
    cx_r = (wr * xs).sum((2, 3)) / sum_m / wn
    cy_t = (wt * ys).sum((2, 3)) / sum_m / hn
    cx_t = (wt * xs).sum((2, 3)) / sum_m / wn
    return (cy_r - cy_t).pow(2) + (cx_r - cx_t).pow(2)


def _fg_quantile_inverse_weights(
    target: torch.Tensor,
    fg: torch.Tensor,
    *,
    n_bins: int,
    power: float,
) -> torch.Tensor:
    """Batched FG weights: rare target quantile bins get higher weight (GPU vectorized)."""
    b, _c, h, width = target.shape
    n_pix = h * width
    m = (fg.reshape(b, n_pix) > 0.5)
    n_fg = m.sum(dim=1)
    min_fg = max(4, n_bins)
    ok = n_fg >= min_fg

    flat = target.reshape(b, n_pix).float()
    masked = flat.masked_fill(~m, float("nan"))
    q = torch.linspace(0.0, 1.0, n_bins + 1, device=target.device, dtype=torch.float32)
    edges = torch.nanquantile(masked, q, dim=1)
    if edges.shape[0] != b:
        edges = edges.transpose(0, 1).contiguous()

    # Degenerate rows: fall back to min–max span on FG.
    row_min = flat.masked_fill(~m, float("inf")).min(dim=1).values
    row_max = flat.masked_fill(~m, float("-inf")).max(dim=1).values
    fallback = row_min.unsqueeze(1) + (row_max - row_min).unsqueeze(1) * q.unsqueeze(0)
    flat_edges = (edges[:, -1] - edges[:, 0]).abs() < 1e-8
    edges = torch.where(flat_edges.unsqueeze(1), fallback, edges)

    nb = edges.shape[1] - 1
    if nb == 1:
        mid = (edges[:, 0] + edges[:, 1]) * 0.5
        idx = (flat >= mid.unsqueeze(1)).long()
    else:
        inner = edges[:, 1:-1].contiguous()
        idx = torch.searchsorted(inner, flat.contiguous(), right=True).clamp(max=nb - 1)

    idx = idx.clamp(0, nb - 1)
    oh = torch.nn.functional.one_hot(idx, nb).float()
    counts = (oh * m.unsqueeze(-1).float()).sum(dim=1).clamp(min=1.0)
    inv = (1.0 / counts.gather(1, idx)).pow(power)
    inv = inv * m.float()
    inv = inv / inv.sum(dim=1, keepdim=True).clamp(min=1e-6) * n_fg.float().unsqueeze(1)

    weights = torch.where(m, inv, torch.ones_like(inv))
    weights = torch.where(ok.unsqueeze(1), weights, torch.ones_like(weights))
    return weights.reshape(b, 1, h, width).to(dtype=target.dtype)


def _fg_huber_base(
    huber: torch.Tensor,
    fg: torch.Tensor,
    target: torch.Tensor,
    c,
) -> torch.Tensor:
    """FG huber mean; optional quantile inverse blend to upweight rare high bins."""
    n_fg = fg.sum((1, 2, 3)).clamp(min=1.0)
    base_u = (huber * fg).sum((1, 2, 3)) / n_fg
    blend = float(getattr(c, "heatmap_quantile_weight", 0.0))
    if blend <= 0.0:
        return base_u
    n_bins = int(getattr(c, "heatmap_quantile_bins", 10))
    power = float(getattr(c, "heatmap_quantile_power", 0.75))
    pix_w = _fg_quantile_inverse_weights(target, fg, n_bins=n_bins, power=power)
    denom = (fg * pix_w).sum((1, 2, 3)).clamp(min=1e-6)
    base_q = (huber * fg * pix_w).sum((1, 2, 3)) / denom
    mix = min(blend, 1.0)
    return (1.0 - mix) * base_u + mix * base_q


def heatmap_loss_gmax(
    recon: torch.Tensor,
    target: torch.Tensor,
    c,
    ps: float,
    *,
    dynrange_weight: float | None = None,
    lite: bool = False,
    clip_fn=None,
    lap_k_fn=None,
    percentile_fn=None,
) -> torch.Tensor:
    """Per-sample heatmap loss (B,) tuned for Ω/global_max normalization."""
    from experiments.exp043.codes import train_vae_simple as _tr

    if clip_fn is None:
        clip_fn = _tr._clip_recon_heatmap_z
    if lap_k_fn is None:
        lap_k_fn = _tr._lap_k
    if percentile_fn is None:
        percentile_fn = _tr._percentile_along_dim

    recon = clip_fn(recon, c)
    fg, bg_m, bg = _fg_bg_masks(target, c)
    huber_delta = float(getattr(c, "heatmap_huber_delta", 0.2))
    grad_delta = float(getattr(c, "heatmap_grad_huber_delta", 0.1))
    dyn_over_w = float(getattr(c, "heatmap_dynrange_over_weight", 0.5))

    huber = F.huber_loss(recon, target, delta=huber_delta, reduction="none")
    n_fg = fg.sum((1, 2, 3)).clamp(min=1.0)
    base = _fg_huber_base(huber, fg, target, c)
    intens = (target - bg).clamp(min=0)
    peak = (huber * fg * intens).sum((1, 2, 3)) / (intens * fg).sum((1, 2, 3)).clamp(min=1e-6)
    flat_r = (recon * fg + (1 - fg) * recon.amin(dim=(2, 3), keepdim=True)).flatten(1)
    flat_t = (target * fg + (1 - fg) * target.amin(dim=(2, 3), keepdim=True)).flatten(1)
    max_r, max_t = flat_r.max(1).values, flat_t.max(1).values
    dyn_max = F.relu(max_t - max_r).pow(2) + dyn_over_w * F.relu(max_r - max_t).pow(2)
    dw = c.heatmap_dynrange_weight if dynrange_weight is None else dynrange_weight
    flat_fg = fg.view(fg.shape[0], -1)
    pattern_w = float(getattr(c, "heatmap_pattern_weight", 0.0))
    spread_w = float(getattr(c, "heatmap_spread_weight", 0.0))
    hotspot_w = float(getattr(c, "heatmap_hotspot_weight", 0.0))
    centroid_w = float(getattr(c, "heatmap_peak_centroid_weight", 0.0))
    pattern = _fg_pattern_correlation_loss(flat_r, flat_t, flat_fg) if pattern_w > 0 else None
    spread = (
        _fg_spread_loss(recon, target, fg, flat_r, flat_t, percentile_fn)
        if spread_w > 0
        else None
    )
    hotspot = _fg_hotspot_loss(huber, target, fg, bg) if hotspot_w > 0 else None
    centroid = (
        _fg_peak_centroid_loss(recon, target, fg, percentile_fn)
        if centroid_w > 0
        else None
    )

    def _add_aux(out: torch.Tensor) -> torch.Tensor:
        if pattern_w > 0 and pattern is not None:
            out = out + pattern_w * pattern
        if spread_w > 0 and spread is not None:
            out = out + spread_w * spread
        if hotspot_w > 0 and hotspot is not None:
            out = out + hotspot_w * hotspot
        if centroid_w > 0 and centroid is not None:
            out = out + centroid_w * centroid
        return out

    if lite:
        return _add_aux(base + c.heatmap_peak_weight * peak + dw * dyn_max)

    fg_dx = fg[:, :, :, 1:] * fg[:, :, :, :-1]
    fg_dy = fg[:, :, 1:, :] * fg[:, :, :-1, :]
    gx = F.huber_loss(
        recon[:, :, :, 1:] - recon[:, :, :, :-1],
        target[:, :, :, 1:] - target[:, :, :, :-1],
        delta=grad_delta,
        reduction="none",
    )
    gy = F.huber_loss(
        recon[:, :, 1:, :] - recon[:, :, :-1, :],
        target[:, :, 1:, :] - target[:, :, :-1, :],
        delta=grad_delta,
        reduction="none",
    )
    grad = (gx * fg_dx).sum((1, 2, 3)) / fg_dx.sum((1, 2, 3)).clamp(min=1.0)
    grad += (gy * fg_dy).sum((1, 2, 3)) / fg_dy.sum((1, 2, 3)).clamp(min=1.0)
    lap_k = lap_k_fn(recon.device, recon.dtype)
    stacked = torch.cat([recon, target], dim=0)
    lap_both = F.conv2d(stacked, lap_k, padding=1)
    lap_r, lap_t = lap_both.chunk(2, dim=0)
    lap = ((lap_r - lap_t).pow(2) * fg).sum((1, 2, 3)) / n_fg
    n_bg = bg_m.sum((1, 2, 3)).clamp(min=1.0)
    contrast = F.relu(
        (recon * bg_m).sum((1, 2, 3)) / n_bg
        + c.heatmap_contrast_margin
        - (recon * fg).sum((1, 2, 3)) / n_fg
    )
    bg_h = (huber * bg_m).sum((1, 2, 3)) / n_bg
    p95_r = percentile_fn(flat_r, 0.95, dim=1)
    p95_t = percentile_fn(flat_t, 0.95, dim=1)
    dyn_p95 = F.relu(p95_t - p95_r).pow(2) + dyn_over_w * F.relu(p95_r - p95_t).pow(2)
    dyn = dyn_max + 0.5 * dyn_p95
    return _add_aux(
        base
        + c.heatmap_peak_weight * peak
        + c.heatmap_grad_weight * grad
        + ps
        * (
            c.heatmap_lap_weight * lap
            + c.heatmap_contrast_weight * contrast
            + c.heatmap_bg_weight * bg_h
            + dw * dyn
        )
    )


def cross_freq_heatmap_loss_gmax(
    model,
    z: torch.Tensor,
    K: torch.Tensor,
    pi_alt: torch.Tensor,
    hm_alt: torch.Tensor,
    c,
    ps: float,
    *,
    dynrange_weight: float | None = None,
    occupancy: torch.Tensor | None = None,
    heatmap_skips: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Decode same z at alternate PI_freq; full gmax heatmap loss by default."""
    base = getattr(model, "_orig_mod", model)
    dev = z.device
    hm_alt = hm_alt.to(dev, non_blocking=c.is_cuda())
    if hm_alt.dim() == 3:
        hm_alt = hm_alt.unsqueeze(1)
    rh, _, _ = base.decode(
        z, K, pi_alt, occupancy=occupancy, heatmap_skips=heatmap_skips,
    )
    use_full = bool(getattr(c, "cross_freq_full_loss", True))
    return heatmap_loss_gmax(
        rh,
        hm_alt,
        c,
        ps,
        dynrange_weight=dynrange_weight,
        lite=not use_full,
    ).mean()
