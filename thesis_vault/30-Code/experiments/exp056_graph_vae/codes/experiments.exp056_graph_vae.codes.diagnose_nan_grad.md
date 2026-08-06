---
title: diagnose_nan_grad
type: code
path: experiments/exp056_graph_vae/codes/diagnose_nan_grad.py
group: experiments/exp056_graph_vae/codes
experiment: exp056_graph_vae
loc: 184
tags: [code, exp056_graph_vae]
---

# diagnose_nan_grad

> Isolate which loss term / layer produces non-finite gradients (exp052).

**Source:** `experiments/exp056_graph_vae/codes/diagnose_nan_grad.py` · 184 lines
**Experiment:** [[exp056_graph_vae]]

## Functions

- **`_load(c: Config, ckpt: Path, device: str)`**
- **`_build_terms(model, batch, c: Config, *, epoch: int, beta: float, imp_log_std: float)`**
- **`_grad_probe(term_name: str, term: torch.Tensor, model)`**
- **`main()`**

## Imports

- [[experiments.exp056_graph_vae.codes.__init__]]
- [[experiments.exp056_graph_vae.codes.dataloader_multifreq]]
- [[experiments.exp056_graph_vae.codes.run_epoch_encode]]
- [[experiments.exp056_graph_vae.codes.train_core]]
- [[experiments.exp056_graph_vae.codes.train_vae_simple]]
- [[experiments.exp056_graph_vae.codes.training_guard]]

## External dependencies

`torch`
