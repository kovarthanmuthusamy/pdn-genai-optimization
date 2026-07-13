"""Finite-loss guards for exp053 training (NaN / Inf prevention)."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def recon_z_soft_cap(c: Any) -> float | None:
    cap = getattr(c, "training_recon_z_soft_cap", None)
    if cap is not None:
        return float(cap)
    bundle = getattr(c, "_norm_stats", None)
    if bundle is not None:
        z_max = getattr(bundle.heatmap, "z_max", None)
        if z_max is not None:
            mult = float(getattr(c, "training_recon_z_cap_mult", 1.25))
            return float(z_max) * mult
    return None


def prepare_recon_for_loss(rh: Tensor, c: Any) -> Tensor:
    """Sanitize recon z before loss; soft-cap extreme values (loss path only)."""
    rh = torch.nan_to_num(rh, nan=0.0, posinf=0.0, neginf=0.0)
    cap = recon_z_soft_cap(c)
    if cap is not None:
        lo = float(getattr(c, "training_recon_z_soft_floor", -6.0))
        rh = rh.clamp(min=lo, max=cap)
    return rh


def clamp_logvar(logvar: Tensor, c: Any) -> Tensor:
    mx = float(getattr(c, "training_logvar_clamp_max", 10.0))
    mn = float(getattr(c, "training_logvar_clamp_min", -10.0))
    return logvar.clamp(min=mn, max=mx)


def sanitize_scalar_loss(t: Tensor, *, cap: float = 1e4) -> Tensor:
    t = torch.nan_to_num(t, nan=0.0, posinf=cap, neginf=0.0)
    return t.clamp(max=cap)


def sanitize_loss_dict(losses: dict[str, Tensor], c: Any) -> dict[str, Tensor]:
    cap = float(getattr(c, "training_loss_finite_cap", 1e4))
    for key, val in list(losses.items()):
        if torch.is_tensor(val):
            losses[key] = sanitize_scalar_loss(val, cap=cap)
    return losses


def is_finite_loss(t: Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


def has_finite_grads(params) -> bool:
    for param in params:
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            return False
    return True



def find_nonfinite_grad_names(
    model,
    physics=None,
    *,
    max_names: int = 5,
) -> list[str]:
    """Return parameter names whose .grad contains NaN/Inf."""
    bad: list[str] = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            bad.append(name)
            if len(bad) >= max_names:
                return bad
    if physics is not None:
        for name, param in physics.named_parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                bad.append(f"physics.{name}")
                if len(bad) >= max_names:
                    break
    return bad



def sanitize_nonfinite_grads(params) -> int:
    """Zero NaN/Inf gradients in-place; returns count of sanitized tensors."""
    n = 0
    for param in params:
        if param.grad is None:
            continue
        if torch.isfinite(param.grad).all():
            continue
        param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
        n += 1
    return n


def find_nonfinite_grad_names_from_params(params, *, max_names: int = 5) -> list[str]:
    bad: list[str] = []
    for param in params:
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            bad.append(getattr(param, "_exp052_name", "<unnamed>"))
            if len(bad) >= max_names:
                break
    return bad


def safe_train_step(
    total_loss: Tensor,
    params,
    optimizer,
    scaler=None,
    *,
    model=None,
    physics=None,
    grad_clip: float = 1.0,
    skip_nonfinite: bool = True,
    use_scaler: bool = True,
    sanitize_grads: bool = False,
) -> tuple[bool, str, list[str]]:
    """Run backward + optimizer step; skip batch when loss/grads are non-finite."""
    if skip_nonfinite and not is_finite_loss(total_loss):
        optimizer.zero_grad(set_to_none=True)
        return False, "nonfinite_loss", []

    if use_scaler and scaler is not None and scaler.is_enabled():
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
        if not torch.isfinite(grad_norm) or not has_finite_grads(params):
            bad = find_nonfinite_grad_names(model, physics, max_names=5) if model is not None else find_nonfinite_grad_names_from_params(params, max_names=5)
            if sanitize_grads:
                sanitize_nonfinite_grads(params)
                grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
            elif skip_nonfinite:
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                return False, "nonfinite_grad", bad
        scaler.step(optimizer)
        scaler.update()
        return True, "ok", []

    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
    if not torch.isfinite(grad_norm) or not has_finite_grads(params):
        bad = find_nonfinite_grad_names(model, physics, max_names=5) if model is not None else find_nonfinite_grad_names_from_params(params, max_names=5)
        if sanitize_grads:
            sanitize_nonfinite_grads(params)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
        elif skip_nonfinite:
            optimizer.zero_grad(set_to_none=True)
            return False, "nonfinite_grad", bad
    optimizer.step()
    return True, "ok", []
