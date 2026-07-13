"""Heatmap losses: Pearson+grad tier_a + log1p peak/valley extrema."""

from __future__ import annotations
from typing import Literal
import torch
import torch.nn.functional as F
from experiments.exp057_structured_graph.codes.spatial_metrics import pearson_fg_loss
from src_vae.others.norm_stats import NormStatsBundle

ExtremaSide = Literal["peak", "valley"]
_SOBEL: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
_TERMS: dict[ExtremaSide, tuple[tuple[str, str, str], ...]] = {
    "peak": (
        ("hotspot", "heatmap_peak_hotspot_weight", "spot"),
        ("max", "heatmap_peak_max_log1p_weight", "extrema"),
        ("centroid", "heatmap_peak_centroid_weight", "centroid"),
        ("topregion", "heatmap_peak_topregion_weight", "topregion"),
    ),
    "valley": (
        ("coldspot", "heatmap_valley_coldspot_weight", "spot"),
        ("min", "heatmap_valley_min_log1p_weight", "extrema"),
        ("centroid", "heatmap_valley_centroid_weight", "centroid"),
        ("topregion", "heatmap_valley_topregion_weight", "topregion"),
    ),
}


def _sobel(hm):
    key = (str(hm.device), str(hm.dtype))
    if key not in _SOBEL:
        gx = torch.tensor([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]], dtype=hm.dtype, device=hm.device)
        gy = torch.tensor([[[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]], dtype=hm.dtype, device=hm.device)
        _SOBEL[key] = (gx, gy)
    gx, gy = _SOBEL[key]
    return F.conv2d(hm, gx, padding=1), F.conv2d(hm, gy, padding=1)


def _finite(t, *, posinf=1e4, neginf=0.0):
    return torch.nan_to_num(t, nan=0.0, posinf=posinf, neginf=neginf)


def grad_vector_field_loss(recon, target, fg, c):
    gr_x, gr_y = _sobel(recon)
    gt_x, gt_y = _sobel(target)
    min_mag, delta = float(c.heatmap_grad_direction_min_mag), float(c.heatmap_grad_huber_delta)
    eps = 1e-6
    mag_t = torch.sqrt(gt_x.pow(2) + gt_y.pow(2) + eps)
    active = fg * (mag_t > min_mag).float()
    n = fg.sum((1, 2, 3)).clamp(min=1.0)
    vec = (F.huber_loss(gr_x, gt_x, delta=delta, reduction="none") + F.huber_loss(gr_y, gt_y, delta=delta, reduction="none"))
    vec_loss = (vec * fg).sum((1, 2, 3)) / n
    mag_r = torch.sqrt(gr_x.pow(2) + gr_y.pow(2) + eps)
    cos = (gr_x * gt_x + gr_y * gt_y) / (mag_r * mag_t)
    dir_loss = (1.0 - cos.clamp(-1., 1.)).mul(active).sum((1, 2, 3)) / active.sum((1, 2, 3)).clamp(min=1.0)
    return _finite(float(c.heatmap_grad_vector_weight) * vec_loss + float(c.heatmap_grad_direction_weight) * dir_loss)


def _robust_log1p_maps(recon, target, c, *, pi_freq):
    bundle: NormStatsBundle | None = getattr(c, "_norm_stats", None)
    if bundle and bundle.heatmap.is_robust_per_mhz() and pi_freq is not None:
        hm = bundle.heatmap
        rows_r, rows_t = [], []
        for i in range(recon.shape[0]):
            b = hm.bin_stats(pi_norm=pi_freq[i : i + 1])
            rows_r.append(recon[i : i + 1] * b.iqr + b.median)
            rows_t.append(target[i : i + 1] * b.iqr + b.median)
        return torch.cat(rows_r, 0), torch.cat(rows_t, 0)
    if bundle:
        b = bundle.heatmap.bin_stats()
        return recon * b.log_std + b.log_mean, target * b.log_std + b.log_mean
    return recon * c._hm_log_std + c._hm_log_mean, target * c._hm_log_std + c._hm_log_mean


def _prep_log1p_fg(recon, target, c, *, pi_freq):
    bg = c.background_value + 0.5
    fg = (target > bg).float()
    return *_robust_log1p_maps(recon, target, c, pi_freq=pi_freq), fg, target, bg


def _fg_percentile(flat, q, dim=1):
    if hasattr(torch, "quantile"):
        return torch.quantile(flat, q, dim=dim, keepdim=True)
    k = max(1, int(round(q * (flat.shape[dim] - 1))))
    return flat.kthvalue(k, dim=dim).values.unsqueeze(dim)


def _salience_log1p_l1(recon_lp, tgt_lp, salience):
    denom = salience.amax((2, 3), keepdim=True).clamp(min=1e-6)
    w = (salience / denom).pow(2)
    diff = (recon_lp - tgt_lp).abs()
    return (diff * w).sum((1, 2, 3)) / w.sum((1, 2, 3)).clamp(min=1e-6)


def _fg_extrema_log1p(recon_lp, tgt_lp, fg, *, side):
    lo, hi = torch.finfo(recon_lp.dtype).min, torch.finfo(recon_lp.dtype).max
    sentinel = lo if side == "peak" else hi
    r = recon_lp.masked_fill(fg < 0.5, sentinel)
    t = tgt_lp.masked_fill(fg < 0.5, sentinel)
    fn = torch.amax if side == "peak" else torch.amin
    return (fn(r, (2, 3)) - fn(t, (2, 3))).abs().view(-1)


def _centroid_log1p(recon_lp, tgt_lp, fg, *, side, c):
    b, _, h, w = recon_lp.shape
    flat_t = (tgt_lp * fg).flatten(1)
    if side == "peak":
        thr = _fg_percentile(flat_t, float(c.heatmap_peak_centroid_top_q)).view(b, 1, 1, 1)
        mask = (tgt_lp >= thr).float() * fg
        wr, wt = recon_lp.clamp(min=0) * mask, tgt_lp * mask
        sum_m = mask.sum((2, 3)).clamp(min=1e-6)
    else:
        thr = _fg_percentile(flat_t, float(c.heatmap_valley_centroid_bottom_q)).view(b, 1, 1, 1)
        mask = (tgt_lp <= thr).float() * fg
        peak = (tgt_lp * fg).amax((2, 3), keepdim=True)
        dark = (peak - tgt_lp).clamp(min=0) * mask
        wr, wt, sum_m = recon_lp * dark, tgt_lp * dark, dark.sum((2, 3)).clamp(min=1e-6)
    ys = torch.arange(h, device=recon_lp.device, dtype=recon_lp.dtype).view(1, 1, h, 1)
    xs = torch.arange(w, device=recon_lp.device, dtype=recon_lp.dtype).view(1, 1, 1, w)
    hn, wn = max(h - 1, 1), max(w - 1, 1)
    cy_r, cx_r = (wr * ys).sum((2, 3)) / sum_m / hn, (wr * xs).sum((2, 3)) / sum_m / wn
    cy_t, cx_t = (wt * ys).sum((2, 3)) / sum_m / hn, (wt * xs).sum((2, 3)) / sum_m / wn
    return ((cy_r - cy_t).pow(2) + (cx_r - cx_t).pow(2)).reshape(-1)


def _topregion_huber(recon_lp, tgt_lp, fg, *, side, c):
    """Huber on log1p amplitude, restricted to top (peak) / bottom (valley) GT percentile FG pixels.

    Cares about both peak/valley *position* (only extreme-region pixels count) and *amplitude*
    (Huber on the log1p physical value). Replaces the centroid position-only term.
    """
    b = recon_lp.shape[0]
    delta = float(getattr(c, "heatmap_topregion_huber_delta", 0.5))
    if side == "peak":
        q = float(getattr(c, "heatmap_peak_topregion_q", 0.95))
        # Fill background with the per-sample min so it never enters the TOP percentile.
        flat_t = (tgt_lp * fg + (1.0 - fg) * tgt_lp.amin((2, 3), keepdim=True)).flatten(1)
        thr = _fg_percentile(flat_t, q).view(b, 1, 1, 1)
        mask = (tgt_lp >= thr).float() * fg
    else:
        q = float(getattr(c, "heatmap_valley_topregion_q", 0.05))
        # Fill background with the per-sample max so it never enters the BOTTOM percentile.
        flat_t = (tgt_lp * fg + (1.0 - fg) * tgt_lp.amax((2, 3), keepdim=True)).flatten(1)
        thr = _fg_percentile(flat_t, q).view(b, 1, 1, 1)
        mask = (tgt_lp <= thr).float() * fg
    hub = F.huber_loss(recon_lp, tgt_lp, delta=delta, reduction="none")
    n = mask.sum((1, 2, 3)).clamp(min=1.0)
    return (hub * mask).sum((1, 2, 3)) / n


def _spot_salience(target_z, fg, bg, *, side):
    intens = (target_z - bg).clamp(min=0) * fg
    if side == "peak":
        return intens
    peak = intens.amax((2, 3), keepdim=True).clamp(min=1e-6)
    return (peak - intens).clamp(min=0)


def _term(kind, recon_lp, tgt_lp, fg, target_z, bg, *, side, c):
    if kind == "spot":
        return _salience_log1p_l1(recon_lp, tgt_lp, _spot_salience(target_z, fg, bg, side=side))
    if kind == "extrema":
        return _fg_extrema_log1p(recon_lp, tgt_lp, fg, side=side)
    if kind == "centroid":
        return _centroid_log1p(recon_lp, tgt_lp, fg, side=side, c=c)
    if kind == "topregion":
        return _topregion_huber(recon_lp, tgt_lp, fg, side=side, c=c)
    raise ValueError(kind)


def _bundle(recon_lp, tgt_lp, fg, target_z, bg, c, *, side):
    out = {}
    for name, weight_attr, kind in _TERMS[side]:
        if float(getattr(c, weight_attr, 0)) <= 0:
            continue
        cap = 4.0 if kind in ("centroid", "topregion") else 1e4
        out[name] = _finite(_term(kind, recon_lp, tgt_lp, fg, target_z, bg, side=side, c=c), posinf=cap)
    return out


def _total(terms, c, *, side, recon):
    if not terms:
        return recon.new_zeros(recon.shape[0])
    total = recon.new_zeros(recon.shape[0])
    for name, weight_attr, _ in _TERMS[side]:
        if name in terms:
            total = total + float(getattr(c, weight_attr)) * terms[name].reshape(recon.shape[0])
    return _finite(total)


def heatmap_extrema_log1p_both(recon, target, c, *, pi_freq=None, sides=("peak", "valley")):
    prep = _prep_log1p_fg(recon, target, c, pi_freq=pi_freq)
    recon_lp, tgt_lp, fg, target_z, bg = prep
    out = {}
    for side in sides:
        terms = _bundle(recon_lp, tgt_lp, fg, target_z, bg, c, side=side)
        if terms:
            out[side] = (_total(terms, c, side=side, recon=recon), terms)
    return out


def heatmap_loss_pearson_grad(recon, target, c, *, lite=False):
    bg = c.background_value + 0.5
    fg = (target > bg).float()
    pearson = pearson_fg_loss(recon, target, fg) * float(c.heatmap_pearson_weight)
    if lite:
        return _finite(pearson, posinf=2.0, neginf=0.0)
    return _finite(pearson + grad_vector_field_loss(recon, target, fg, c))
