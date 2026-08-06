---
title: diagnose_nan_grad
type: code
path: experiments/exp052_unbounded_pearson/codes/diagnose_nan_grad.py
group: experiments/exp052_unbounded_pearson/codes
experiment: exp052_unbounded_pearson
loc: 209
tags: [code, exp052_unbounded_pearson]
---

# diagnose_nan_grad

> Isolate which loss term / layer produces non-finite gradients (exp052).

**Source:** `experiments/exp052_unbounded_pearson/codes/diagnose_nan_grad.py` · 209 lines
**Experiment:** [[exp052_unbounded_pearson]]

## Functions

- **`_load(c: Config, ckpt: Path, device: str)`**
- **`_build_terms(model, physics, batch, c: Config, *, epoch: int, beta: float, imp_log_std: float)`**
- **`_grad_probe(term_name: str, term: torch.Tensor, model, physics)`**
- **`main()`**

## Imports

- [[experiments.exp038_true_multi.codes.physics_loss]]
- [[experiments.exp038_true_multi.codes.train_vae_simple]]
- [[experiments.exp052_unbounded_pearson.codes.__init__]]
- [[experiments.exp052_unbounded_pearson.codes.dataloader_multifreq]]
- [[experiments.exp052_unbounded_pearson.codes.heatmap_peak_losses]]
- [[experiments.exp052_unbounded_pearson.codes.mhz_loss_weight]]
- [[experiments.exp052_unbounded_pearson.codes.run_epoch_encode]]
- [[experiments.exp052_unbounded_pearson.codes.train_vae_simple]]
- [[experiments.exp052_unbounded_pearson.codes.training_guard]]

## External dependencies

`torch`
