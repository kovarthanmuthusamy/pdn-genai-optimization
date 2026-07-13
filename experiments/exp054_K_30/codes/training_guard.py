"""Finite-loss / NaN guards for exp054 training."""

from __future__ import annotations
from typing import Any
import torch
from torch import Tensor


def recon_z_soft_cap(c: Any) -> float | None:
    cap = getattr(c, "training_recon_z_soft_cap", None)
    if cap is not None:
        return float(cap)
    bundle = getattr(c, "_norm_stats", None)
    z_max = getattr(getattr(bundle, "heatmap", None), "z_max", None) if bundle else None
    if z_max is not None:
        return float(z_max) * float(c.training_recon_z_cap_mult)
    return None


def prepare_recon_for_loss(rh: Tensor, c: Any) -> Tensor:
    rh = torch.nan_to_num(rh, nan=0.0, posinf=0.0, neginf=0.0)
    cap = recon_z_soft_cap(c)
    return rh.clamp(min=float(c.training_recon_z_soft_floor), max=cap) if cap is not None else rh


def clamp_logvar(logvar: Tensor, c: Any) -> Tensor:
    return logvar.clamp(min=float(c.training_logvar_clamp_min), max=float(c.training_logvar_clamp_max))


def sanitize_loss_dict(losses: dict[str, Tensor], c: Any) -> dict[str, Tensor]:
    cap = float(c.training_loss_finite_cap)
    for k, v in list(losses.items()):
        if torch.is_tensor(v):
            losses[k] = torch.nan_to_num(v, nan=0.0, posinf=cap, neginf=0.0).clamp(max=cap)
    return losses


def _grads_finite(params) -> bool:
    return all(p.grad is None or torch.isfinite(p.grad).all() for p in params)


def _bad_grad_names(model, params, *, max_names=5) -> list[str]:
    bad = []
    if model is not None:
        src = ((n, p) for n, p in model.named_parameters())
    else:
        src = ((getattr(p, "_exp052_name", "?"), p) for p in params)
    for name, p in src:
        if p.grad is not None and not torch.isfinite(p.grad).all():
            bad.append(name)
            if len(bad) >= max_names:
                break
    return bad


def _sanitize_grads(params) -> None:
    for p in params:
        if p.grad is not None and not torch.isfinite(p.grad).all():
            p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)


def safe_train_step(total_loss, params, optimizer, scaler=None, *, model=None, grad_clip=1.0,
                    skip_nonfinite=True, use_scaler=True, sanitize_grads=False):
    if skip_nonfinite and not bool(torch.isfinite(total_loss).all()):
        optimizer.zero_grad(set_to_none=True)
        return False, "nonfinite_loss", []

    def _clip_ok():
        gn = torch.nn.utils.clip_grad_norm_(params, grad_clip)
        if torch.isfinite(gn) and _grads_finite(params):
            return True, [], gn
        bad = _bad_grad_names(model, params)
        if sanitize_grads:
            _sanitize_grads(params)
            gn = torch.nn.utils.clip_grad_norm_(params, grad_clip)
            return torch.isfinite(gn) and _grads_finite(params), bad, gn
        return False, bad, gn

    if use_scaler and scaler is not None and scaler.is_enabled():
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        ok, bad, _ = _clip_ok()
        if not ok:
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            return False, "nonfinite_grad", bad
        scaler.step(optimizer)
        scaler.update()
        return True, "ok", []

    total_loss.backward()
    ok, bad, _ = _clip_ok()
    if not ok:
        optimizer.zero_grad(set_to_none=True)
        return False, "nonfinite_grad", bad
    optimizer.step()
    return True, "ok", []


find_nonfinite_grad_names = lambda model, max_names=5: _bad_grad_names(model, [], max_names=max_names)
