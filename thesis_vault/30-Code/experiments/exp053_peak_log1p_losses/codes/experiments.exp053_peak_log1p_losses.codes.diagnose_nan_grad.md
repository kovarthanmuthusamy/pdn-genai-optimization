---
title: diagnose_nan_grad
type: code
path: experiments/exp053_peak_log1p_losses/codes/diagnose_nan_grad.py
group: experiments/exp053_peak_log1p_losses/codes
experiment: exp053_peak_log1p_losses
loc: 209
tags: [code, exp053_peak_log1p_losses]
---

# diagnose_nan_grad

> Isolate which loss term / layer produces non-finite gradients (exp052).

**Source:** `experiments/exp053_peak_log1p_losses/codes/diagnose_nan_grad.py` · 209 lines
**Experiment:** [[exp053_peak_log1p_losses]]

## Functions

- **`_load(c: Config, ckpt: Path, device: str)`**
- **`_build_terms(model, physics, batch, c: Config, *, epoch: int, beta: float, imp_log_std: float)`**
- **`_grad_probe(term_name: str, term: torch.Tensor, model, physics)`**
- **`main()`**

## Imports

- [[experiments.exp038_true_multi.codes.physics_loss]]
- [[experiments.exp038_true_multi.codes.train_vae_simple]]
- [[experiments.exp053_peak_log1p_losses.codes.__init__]]
- [[experiments.exp053_peak_log1p_losses.codes.dataloader_multifreq]]
- [[experiments.exp053_peak_log1p_losses.codes.heatmap_peak_losses]]
- [[experiments.exp053_peak_log1p_losses.codes.mhz_loss_weight]]
- [[experiments.exp053_peak_log1p_losses.codes.run_epoch_encode]]
- [[experiments.exp053_peak_log1p_losses.codes.train_vae_simple]]
- [[experiments.exp053_peak_log1p_losses.codes.training_guard]]

## External dependencies

`torch`
