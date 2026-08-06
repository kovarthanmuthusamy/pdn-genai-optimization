---
title: diagnose_nan_grad
type: code
path: experiments/exp054_K_30/codes/diagnose_nan_grad.py
group: experiments/exp054_K_30/codes
experiment: exp054_K_30
loc: 184
tags: [code, exp054_K_30]
---

# diagnose_nan_grad

> Isolate which loss term / layer produces non-finite gradients (exp052).

**Source:** `experiments/exp054_K_30/codes/diagnose_nan_grad.py` · 184 lines
**Experiment:** [[exp054_K_30]]

## Functions

- **`_load(c: Config, ckpt: Path, device: str)`**
- **`_build_terms(model, batch, c: Config, *, epoch: int, beta: float, imp_log_std: float)`**
- **`_grad_probe(term_name: str, term: torch.Tensor, model)`**
- **`main()`**

## Imports

- [[experiments.exp054_K_30.codes.__init__]]
- [[experiments.exp054_K_30.codes.dataloader_multifreq]]
- [[experiments.exp054_K_30.codes.run_epoch_encode]]
- [[experiments.exp054_K_30.codes.train_core]]
- [[experiments.exp054_K_30.codes.train_vae_simple]]
- [[experiments.exp054_K_30.codes.training_guard]]

## External dependencies

`torch`
