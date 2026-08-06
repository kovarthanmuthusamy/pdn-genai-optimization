---
title: diagnose_nan_grad
type: code
path: experiments/exp058_asymmetric_kl/codes/diagnose_nan_grad.py
group: experiments/exp058_asymmetric_kl/codes
experiment: exp058_asymmetric_kl
loc: 184
tags: [code, exp058_asymmetric_kl, uncommitted]
---

# diagnose_nan_grad

> Isolate which loss term / layer produces non-finite gradients (exp052).

**Source:** `experiments/exp058_asymmetric_kl/codes/diagnose_nan_grad.py` · 184 lines
**Git:** uncommitted — not yet tracked
**Experiment:** [[exp058_asymmetric_kl]]

## Functions

- **`_load(c: Config, ckpt: Path, device: str)`**
- **`_build_terms(model, batch, c: Config, *, epoch: int, beta: float, imp_log_std: float)`**
- **`_grad_probe(term_name: str, term: torch.Tensor, model)`**
- **`main()`**

## Imports

- [[experiments.exp058_asymmetric_kl.codes.__init__]]
- [[experiments.exp058_asymmetric_kl.codes.dataloader_multifreq]]
- [[experiments.exp058_asymmetric_kl.codes.run_epoch_encode]]
- [[experiments.exp058_asymmetric_kl.codes.train_core]]
- [[experiments.exp058_asymmetric_kl.codes.train_vae_simple]]
- [[experiments.exp058_asymmetric_kl.codes.training_guard]]

## External dependencies

`torch`
