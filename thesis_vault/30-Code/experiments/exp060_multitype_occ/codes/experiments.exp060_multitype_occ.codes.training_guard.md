---
title: training_guard
type: code
path: experiments/exp060_multitype_occ/codes/training_guard.py
group: experiments/exp060_multitype_occ/codes
experiment: exp060_multitype_occ
loc: 101
tags: [code, exp060_multitype_occ, uncommitted]
---

# training_guard

> Finite-loss / NaN guards for exp055 training.

**Source:** `experiments/exp060_multitype_occ/codes/training_guard.py` · 101 lines
**Git:** uncommitted — not yet tracked
**Experiment:** [[exp060_multitype_occ]]

## Functions

- **`recon_z_soft_cap(c: Any)`**
- **`prepare_recon_for_loss(rh: Tensor, c: Any)`**
- **`clamp_logvar(logvar: Tensor, c: Any)`**
- **`sanitize_loss_dict(losses: dict[str, Tensor], c: Any)`**
- **`_grads_finite(params)`**
- **`_bad_grad_names(model, params, *, max_names=5)`**
- **`_sanitize_grads(params)`**
- **`safe_train_step(total_loss, params, optimizer, scaler=None, *, model=None, grad_clip=1.0, skip_nonfinite=True, use_scaler=True, sanitize_grads=False)`**

## Imported by

- [[experiments.exp060_multitype_occ.codes.run_epoch_encode]]
- [[experiments.exp060_multitype_occ.codes.train_vae_simple]]

## External dependencies

`torch`
