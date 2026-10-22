"""Heatmap loss terms and on-board map metrics. Every term reads the board pixels only.

    heatmap_loss = pearson_w * (1 - Pearson)           pattern
                 + level_w   * (peak level error)^2    amplitude (the pattern terms are scale-free)
                 + overlap_w * (1 - soft overlap)      where the heat is
                 + gradient field (vector + direction) edges; skipped in the "lite" form used
                                                       for the cross-frequency decode

The soft-argmax peak location term is added separately in ``objective.vae_loss``.

Usage (import only):
pearson_fg(recon, target)                       (B,) on-board Pearson correlation in [-1, 1]
pearson_fg_loss(recon, target, fg)              (B,) 1 - Pearson over ``fg``
peak_location_loss(recon, target, fg, bg, T)    (B,) squared distance of the soft-argmax peaks
heatmap_level_loss(recon, target, fg)           (B,) squared error of the on-board peak level
heatmap_region_overlap_loss(recon, target, fg)  (B,) 1 - histogram intersection of softmaxed maps
grad_vector_field_loss(recon, target, fg, c)    (B,) Sobel-gradient vector + direction match
heatmap_loss(recon, target, c, lite=False)      (B,) the weighted sum above
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from thesis_code.common.board import board_fg
from thesis_code.s06_losses.guards import finite

_SOBEL: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}


# ── Metrics ──────────────────────────────────────────────────────────────────

def pearson_fg(recon: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample (B,) Pearson correlation over the board pixels, in [-1, 1]."""
    fg = board_fg(target)
    flat_r, flat_t = (recon * fg).flatten(1), (target * fg).flatten(1)
    rc, tc = flat_r - flat_r.mean(1, keepdim=True), flat_t - flat_t.mean(1, keepdim=True)
    den = (rc.pow(2).sum(1).sqrt() * tc.pow(2).sum(1).sqrt()).clamp(min=eps)
    return ((rc * tc).sum(1) / den).clamp(-1.0, 1.0)


def pearson_fg_loss(recon: torch.Tensor, target: torch.Tensor, fg: torch.Tensor, *, eps: float = 1e-4) -> torch.Tensor:
    """Per-sample (B,) ``1 - Pearson`` over ``fg`` (whole map if a sample has < 2 board pixels)."""
    flat_r, flat_t = (recon * fg).flatten(1), (target * fg).flatten(1)
    if (fg.sum((1, 2, 3)).clamp(min=1.0) < 2).any():
        flat_r, flat_t = recon.flatten(1), target.flatten(1)
    rc, tc = flat_r - flat_r.mean(1, keepdim=True), flat_t - flat_t.mean(1, keepdim=True)
    den = (rc.pow(2).sum(1).sqrt() * tc.pow(2).sum(1).sqrt()).clamp(min=eps)
    return torch.nan_to_num(1.0 - (rc * tc).sum(1) / den, nan=2.0, posinf=2.0, neginf=0.0).clamp(-1.0, 1.0)


def _soft_peak_coords(hm: torch.Tensor, fg: torch.Tensor, bg: float, temperature: float):
    """Soft-argmax (row, col) of the peak, normalized to [0, 1]; weights ``(hm - bg)`` clipped at 0."""
    b, _, h, w = hm.shape
    logits = ((hm - bg).clamp(min=0) * fg).view(b, -1) / max(float(temperature), 1e-3)
    wmap = torch.softmax(logits, dim=1)
    ys = torch.arange(h, device=hm.device, dtype=hm.dtype).view(1, h, 1).expand(b, h, w).reshape(b, -1)
    xs = torch.arange(w, device=hm.device, dtype=hm.dtype).view(1, 1, w).expand(b, h, w).reshape(b, -1)
    return (wmap * ys).sum(1) / max(h - 1, 1), (wmap * xs).sum(1) / max(w - 1, 1)


def peak_location_loss(recon, target, fg, bg: float, temperature: float = 0.05) -> torch.Tensor:
    """Per-sample (B,) squared distance between the soft-argmax peaks of ``recon`` and ``target``."""
    ry, rx = _soft_peak_coords(recon, fg, bg, temperature)
    ty, tx = _soft_peak_coords(target, fg, bg, temperature)
    return torch.nan_to_num((ry - ty).pow(2) + (rx - tx).pow(2), nan=0.0, posinf=4.0, neginf=0.0)


# ── Loss terms ───────────────────────────────────────────────────────────────

def _sobel(hm: torch.Tensor):
    key = (str(hm.device), str(hm.dtype))
    if key not in _SOBEL:
        gx = torch.tensor([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]], dtype=hm.dtype, device=hm.device)
        gy = torch.tensor([[[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]], dtype=hm.dtype, device=hm.device)
        _SOBEL[key] = (gx, gy)
    gx, gy = _SOBEL[key]
    return F.conv2d(hm, gx, padding=1), F.conv2d(hm, gy, padding=1)


def grad_vector_field_loss(recon, target, fg, c) -> torch.Tensor:
    """Per-sample (B,) Huber on the Sobel gradient vectors plus ``1 - cos`` on their directions,
    the latter only where the target has an edge (``|grad| > heatmap_grad_direction_min_mag``)."""
    gr_x, gr_y = _sobel(recon)
    gt_x, gt_y = _sobel(target)
    min_mag, delta = float(c.heatmap_grad_direction_min_mag), float(c.heatmap_grad_huber_delta)
    eps = 1e-6
    mag_t = torch.sqrt(gt_x.pow(2) + gt_y.pow(2) + eps)
    active = fg * (mag_t > min_mag).float()
    n = fg.sum((1, 2, 3)).clamp(min=1.0)
    vec = (F.huber_loss(gr_x, gt_x, delta=delta, reduction="none")
           + F.huber_loss(gr_y, gt_y, delta=delta, reduction="none"))
    vec_loss = (vec * fg).sum((1, 2, 3)) / n
    mag_r = torch.sqrt(gr_x.pow(2) + gr_y.pow(2) + eps)
    cos = (gr_x * gt_x + gr_y * gt_y) / (mag_r * mag_t)
    dir_loss = (1.0 - cos.clamp(-1., 1.)).mul(active).sum((1, 2, 3)) / active.sum((1, 2, 3)).clamp(min=1.0)
    return finite(float(c.heatmap_grad_vector_weight) * vec_loss + float(c.heatmap_grad_direction_weight) * dir_loss)


def _board_flat(x: torch.Tensor, fg: torch.Tensor) -> torch.Tensor:
    """(B, H*W) with off-board pixels at the dtype minimum (so max / softmax ignore them)."""
    return torch.where(fg.bool(), x, torch.full_like(x, torch.finfo(x.dtype).min)).flatten(1)


def heatmap_level_loss(recon, target, fg) -> torch.Tensor:
    """Per-sample (B,) squared error of the on-board PEAK LEVEL (normalized units). The pattern
    terms are scale-invariant, so without this nothing constrains amplitude."""
    return (_board_flat(recon, fg).max(1).values - _board_flat(target, fg).max(1).values).pow(2)


def heatmap_region_overlap_loss(recon, target, fg, temp: float = 0.5) -> torch.Tensor:
    """Per-sample (B,) ``1 - histogram intersection`` of temperature-softmaxed board maps: a
    differentiable stand-in for the acceptance gate's top-decile IoU."""
    t = max(float(temp), 1e-6)
    p_r = torch.softmax(_board_flat(recon, fg) / t, dim=1)
    p_t = torch.softmax(_board_flat(target, fg) / t, dim=1)
    return 1.0 - torch.minimum(p_r, p_t).sum(1)


def heatmap_loss(recon, target, c, *, lite: bool = False) -> torch.Tensor:
    """Per-sample (B,) heatmap loss (module docstring); ``lite`` drops the gradient-field term."""
    fg = board_fg(target)
    loss = (c.heatmap_pearson_weight * pearson_fg_loss(recon, target, fg)
            + c.heatmap_level_weight * heatmap_level_loss(recon, target, fg)
            + c.heatmap_softoverlap_weight * heatmap_region_overlap_loss(recon, target, fg, c.heatmap_softoverlap_temp))
    if lite:
        return finite(loss, posinf=2.0, neginf=0.0)
    return finite(loss + grad_vector_field_loss(recon, target, fg, c))
