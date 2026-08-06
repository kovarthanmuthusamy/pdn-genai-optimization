---
title: training_guard
type: code
path: experiments/exp057_structured_graph/codes/training_guard.py
group: experiments/exp057_structured_graph/codes
experiment: exp057_structured_graph
loc: 101
tags: [code, exp057_structured_graph]
---

# training_guard

> Finite-loss / NaN guards for exp055 training.

**Source:** `experiments/exp057_structured_graph/codes/training_guard.py` · 101 lines
**Experiment:** [[exp057_structured_graph]]

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

- [[experiments.exp057_structured_graph.codes.diagnose_nan_grad]]
- [[experiments.exp057_structured_graph.codes.run_epoch_encode]]
- [[experiments.exp057_structured_graph.codes.train_vae_simple]]

## External dependencies

`torch`
