---
title: diagnose_nan_grad
type: code
path: experiments/exp055_hard_occ/codes/diagnose_nan_grad.py
group: experiments/exp055_hard_occ/codes
experiment: exp055_hard_occ
loc: 184
tags: [code, exp055_hard_occ]
---

# diagnose_nan_grad

> Isolate which loss term / layer produces non-finite gradients (exp052).

**Source:** `experiments/exp055_hard_occ/codes/diagnose_nan_grad.py` · 184 lines
**Experiment:** [[exp055_hard_occ]]

## Functions

- **`_load(c: Config, ckpt: Path, device: str)`**
- **`_build_terms(model, batch, c: Config, *, epoch: int, beta: float, imp_log_std: float)`**
- **`_grad_probe(term_name: str, term: torch.Tensor, model)`**
- **`main()`**

## Imports

- [[experiments.exp055_hard_occ.codes.__init__]]
- [[experiments.exp055_hard_occ.codes.dataloader_multifreq]]
- [[experiments.exp055_hard_occ.codes.run_epoch_encode]]
- [[experiments.exp055_hard_occ.codes.train_core]]
- [[experiments.exp055_hard_occ.codes.train_vae_simple]]
- [[experiments.exp055_hard_occ.codes.training_guard]]

## External dependencies

`torch`
