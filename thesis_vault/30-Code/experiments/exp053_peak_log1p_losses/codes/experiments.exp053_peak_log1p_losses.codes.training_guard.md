---
title: training_guard
type: code
path: experiments/exp053_peak_log1p_losses/codes/training_guard.py
group: experiments/exp053_peak_log1p_losses/codes
experiment: exp053_peak_log1p_losses
loc: 166
tags: [code, exp053_peak_log1p_losses]
---

# training_guard

> Finite-loss guards for exp053 training (NaN / Inf prevention).

**Source:** `experiments/exp053_peak_log1p_losses/codes/training_guard.py` · 166 lines
**Experiment:** [[exp053_peak_log1p_losses]]

## Functions

- **`recon_z_soft_cap(c: Any)`**
- **`prepare_recon_for_loss(rh: Tensor, c: Any)`** — Sanitize recon z before loss; soft-cap extreme values (loss path only).
- **`clamp_logvar(logvar: Tensor, c: Any)`**
- **`sanitize_scalar_loss(t: Tensor, *, cap: float=10000.0)`**
- **`sanitize_loss_dict(losses: dict[str, Tensor], c: Any)`**
- **`is_finite_loss(t: Tensor)`**
- **`has_finite_grads(params)`**
- **`find_nonfinite_grad_names(model, physics=None, *, max_names: int=5)`** — Return parameter names whose .grad contains NaN/Inf.
- **`sanitize_nonfinite_grads(params)`** — Zero NaN/Inf gradients in-place; returns count of sanitized tensors.
- **`find_nonfinite_grad_names_from_params(params, *, max_names: int=5)`**
- **`safe_train_step(total_loss: Tensor, params, optimizer, scaler=None, *, model=None, physics=None, grad_clip: float=1.0, skip_nonfinite: bool=True, use_scaler: bool=True, sanitize_grads: bool=False)`** — Run backward + optimizer step; skip batch when loss/grads are non-finite.

## Imported by

- [[experiments.exp053_peak_log1p_losses.codes.diagnose_nan_grad]]
- [[experiments.exp053_peak_log1p_losses.codes.run_epoch_encode]]
- [[experiments.exp053_peak_log1p_losses.codes.train_vae_simple]]

## External dependencies

`torch`
