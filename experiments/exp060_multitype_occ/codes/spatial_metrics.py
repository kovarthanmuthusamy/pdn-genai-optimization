"""FG Pearson + soft extrema location metrics."""

from __future__ import annotations
from typing import Literal
import torch

ExtremaSide = Literal["peak", "valley"]


def pearson_fg(recon, target, bg, *, margin=0.5, eps=1e-6):
    fg = (target > bg + margin).float()
    flat_r, flat_t = (recon * fg).flatten(1), (target * fg).flatten(1)
    r_mean, t_mean = flat_r.mean(1, keepdim=True), flat_t.mean(1, keepdim=True)
    rc, tc = flat_r - r_mean, flat_t - t_mean
    den = (rc.pow(2).sum(1).sqrt() * tc.pow(2).sum(1).sqrt()).clamp(min=eps)
    return ((rc * tc).sum(1) / den).clamp(-1.0, 1.0)


def pearson_fg_loss(recon, target, fg, *, eps=1e-4):
    flat_r, flat_t = (recon * fg).flatten(1), (target * fg).flatten(1)
    if (fg.sum((1, 2, 3)).clamp(min=1.0) < 2).any():
        flat_r, flat_t = recon.flatten(1), target.flatten(1)
    r_mean, t_mean = flat_r.mean(1, keepdim=True), flat_t.mean(1, keepdim=True)
    rc, tc = flat_r - r_mean, flat_t - t_mean
    den = (rc.pow(2).sum(1).sqrt() * tc.pow(2).sum(1).sqrt()).clamp(min=eps)
    return torch.nan_to_num(1.0 - (rc * tc).sum(1) / den, nan=2.0, posinf=2.0, neginf=0.0).clamp(-1.0, 1.0)


def _coord_grid(b, h, w, device, dtype):
    ys = torch.arange(h, device=device, dtype=dtype)
    xs = torch.arange(w, device=device, dtype=dtype)
    return ys.view(1, h, 1).expand(b, h, w).reshape(b, -1), xs.view(1, 1, w).expand(b, h, w).reshape(b, -1)


def _extrema_logits(hm, fg, bg, *, side, temperature):
    b, temp = hm.shape[0], max(float(temperature), 1e-3)
    if side == "peak":
        return ((hm - bg).clamp(min=0) * fg).view(b, -1) / temp
    neg = torch.finfo(hm.dtype).min
    peak = hm.masked_fill(fg < 0.5, neg).amax(dim=(2, 3), keepdim=True)
    return ((peak - hm).clamp(min=0) * fg).view(b, -1) / temp


def _sharp_extrema_coords(hm, fg, bg, *, side, temperature):
    b, _, h, w = hm.shape
    wmap = torch.softmax(_extrema_logits(hm, fg, bg, side=side, temperature=temperature), dim=1)
    gy, gx = _coord_grid(b, h, w, hm.device, hm.dtype)
    hn, wn = max(h - 1, 1), max(w - 1, 1)
    return (wmap * gy).sum(1) / hn, (wmap * gx).sum(1) / wn


def soft_peak_coords(hm, fg, bg):
    b, _, h, w = hm.shape
    wmap = ((hm - bg).clamp(min=0) * fg).view(b, -1)
    wmap = wmap / wmap.sum(1, keepdim=True).clamp(min=1e-8)
    gy, gx = _coord_grid(b, h, w, hm.device, hm.dtype)
    hn, wn = max(h - 1, 1), max(w - 1, 1)
    return (wmap * gy).sum(1) / hn, (wmap * gx).sum(1) / wn


def peak_loc_err(recon, target, bg, *, margin=0.5):
    fg = (target > bg + margin).float()
    th = bg + margin
    ry, rx = soft_peak_coords(recon, fg, th)
    ty, tx = soft_peak_coords(target, fg, th)
    return torch.sqrt((ry - ty).pow(2) + (rx - tx).pow(2))


def extrema_loc_sharp_loss(recon, target, fg, bg, *, side, temperature=0.05):
    ry, rx = _sharp_extrema_coords(recon, fg, bg, side=side, temperature=temperature)
    ty, tx = _sharp_extrema_coords(target, fg, bg, side=side, temperature=temperature)
    return torch.nan_to_num((ry - ty).pow(2) + (rx - tx).pow(2), nan=0.0, posinf=4.0, neginf=0.0)
