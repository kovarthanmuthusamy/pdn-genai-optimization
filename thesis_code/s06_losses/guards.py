"""Numerical guards applied around the loss: clamps, NaN/Inf replacement, caps.

Usage (import only):
finite(t, posinf, neginf)          nan -> 0, +/-inf -> the given values
clamp_logvar(logvar, c)            posterior log-variance into [training_logvar_clamp_min, _max]
prepare_recon_for_loss(rh, c)      heatmap output: non-finite -> 0, floor ``training_recon_z_soft_floor``,
                                   cap = dataset heatmap z_max * ``training_recon_z_cap_mult``
sanitize_loss_dict(losses, c)      every term: nan -> 0, inf -> cap, capped at ``training_loss_finite_cap``
"""
from __future__ import annotations

import torch


def finite(t: torch.Tensor, *, posinf: float = 1e4, neginf: float = 0.0) -> torch.Tensor:
    return torch.nan_to_num(t, nan=0.0, posinf=posinf, neginf=neginf)


def clamp_logvar(logvar: torch.Tensor, c) -> torch.Tensor:
    return logvar.clamp(min=float(c.training_logvar_clamp_min), max=float(c.training_logvar_clamp_max))


def prepare_recon_for_loss(rh: torch.Tensor, c) -> torch.Tensor:
    rh = torch.nan_to_num(rh, nan=0.0, posinf=0.0, neginf=0.0)
    cap = float(c._norm_stats.heatmap.z_max) * float(c.training_recon_z_cap_mult)
    return rh.clamp(min=float(c.training_recon_z_soft_floor), max=cap)


def sanitize_loss_dict(losses: dict[str, torch.Tensor], c) -> dict[str, torch.Tensor]:
    cap = float(c.training_loss_finite_cap)
    for k, v in list(losses.items()):
        if torch.is_tensor(v):
            losses[k] = torch.nan_to_num(v, nan=0.0, posinf=cap, neginf=0.0).clamp(max=cap)
    return losses
