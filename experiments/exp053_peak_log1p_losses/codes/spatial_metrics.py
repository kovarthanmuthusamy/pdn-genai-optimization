"""Foreground spatial metrics — Pearson correlation and soft peak location."""

from __future__ import annotations

import torch


def pearson_fg(
    recon: torch.Tensor,
    target: torch.Tensor,
    bg: float,
    *,
    margin: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-sample Pearson r on foreground pixels; returns (B,) in [-1, 1]."""
    fg = (target > bg + margin).float()
    flat_r = (recon * fg).flatten(1)
    flat_t = (target * fg).flatten(1)
    r_mean = flat_r.mean(1, keepdim=True)
    t_mean = flat_t.mean(1, keepdim=True)
    rc = flat_r - r_mean
    tc = flat_t - t_mean
    num = (rc * tc).sum(1)
    den = (rc.pow(2).sum(1).sqrt() * tc.pow(2).sum(1).sqrt()).clamp(min=eps)
    return (num / den).clamp(-1.0, 1.0)


def pearson_fg_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    fg: torch.Tensor,
    *,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Per-sample loss 1 - Pearson r on fg-masked maps; returns (B,)."""
    flat_r = (recon * fg).flatten(1)
    flat_t = (target * fg).flatten(1)
    n_fg = fg.sum((1, 2, 3)).clamp(min=1.0)
    if (n_fg < 2).any():
        flat_r = recon.flatten(1)
        flat_t = target.flatten(1)
    r_mean = flat_r.mean(1, keepdim=True)
    t_mean = flat_t.mean(1, keepdim=True)
    rc = flat_r - r_mean
    tc = flat_t - t_mean
    num = (rc * tc).sum(1)
    den = (rc.pow(2).sum(1).sqrt() * tc.pow(2).sum(1).sqrt()).clamp(min=eps)
    loss = 1.0 - (num / den).clamp(-1.0, 1.0)
    return torch.nan_to_num(loss, nan=2.0, posinf=2.0, neginf=0.0)


def soft_peak_coords(
    hm: torch.Tensor,
    fg: torch.Tensor,
    bg: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft-argmax peak (y, x) normalized to [0, 1]; returns (B,), (B,)."""
    b, _, h, w = hm.shape
    wmap = ((hm - bg).clamp(min=0) * fg).view(b, -1)
    wmap = wmap / wmap.sum(1, keepdim=True).clamp(min=1e-8)
    ys = torch.arange(h, device=hm.device, dtype=hm.dtype)
    xs = torch.arange(w, device=hm.device, dtype=hm.dtype)
    grid_y = ys.view(1, h, 1).expand(b, h, w).reshape(b, -1)
    grid_x = xs.view(1, 1, w).expand(b, h, w).reshape(b, -1)
    y = (wmap * grid_y).sum(1) / max(h - 1, 1)
    x = (wmap * grid_x).sum(1) / max(w - 1, 1)
    return y, x


def peak_loc_err(
    recon: torch.Tensor,
    target: torch.Tensor,
    bg: float,
    *,
    margin: float = 0.5,
) -> torch.Tensor:
    """L2 distance between soft peak coords (normalized); returns (B,)."""
    fg = (target > bg + margin).float()
    ry, rx = soft_peak_coords(recon, fg, bg + margin)
    ty, tx = soft_peak_coords(target, fg, bg + margin)
    return torch.sqrt((ry - ty).pow(2) + (rx - tx).pow(2))


def peak_loc_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    fg: torch.Tensor,
    bg: float,
) -> torch.Tensor:
    """Squared peak-coordinate error; returns (B,)."""
    ry, rx = soft_peak_coords(recon, fg, bg)
    ty, tx = soft_peak_coords(target, fg, bg)
    loss = (ry - ty).pow(2) + (rx - tx).pow(2)
    return torch.nan_to_num(loss, nan=0.0, posinf=4.0, neginf=0.0)

def peak_loc_sharp_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    fg: torch.Tensor,
    bg: float,
    *,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Low-temperature soft-argmax peak match (closer to hard argmax than peak_loc_loss)."""
    b, _, h, w = recon.shape
    temp = max(float(temperature), 1e-3)

    def _coords(hm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = ((hm - bg).clamp(min=0) * fg).view(b, -1) / temp
        wmap = torch.softmax(logits, dim=1)
        ys = torch.arange(h, device=hm.device, dtype=hm.dtype)
        xs = torch.arange(w, device=hm.device, dtype=hm.dtype)
        grid_y = ys.view(1, h, 1).expand(b, h, w).reshape(b, -1)
        grid_x = xs.view(1, 1, w).expand(b, h, w).reshape(b, -1)
        y = (wmap * grid_y).sum(1) / max(h - 1, 1)
        x = (wmap * grid_x).sum(1) / max(w - 1, 1)
        return y, x

    ry, rx = _coords(recon)
    ty, tx = _coords(target)
    loss = (ry - ty).pow(2) + (rx - tx).pow(2)
    return torch.nan_to_num(loss, nan=0.0, posinf=4.0, neginf=0.0)

