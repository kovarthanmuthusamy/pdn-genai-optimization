---
title: diagnose_nan_grad
type: code
path: experiments/exp057_structured_graph/codes/diagnose_nan_grad.py
group: experiments/exp057_structured_graph/codes
experiment: exp057_structured_graph
loc: 184
tags: [code, exp057_structured_graph]
---

# diagnose_nan_grad

> Isolate which loss term / layer produces non-finite gradients (exp052).

**Source:** `experiments/exp057_structured_graph/codes/diagnose_nan_grad.py` · 184 lines
**Experiment:** [[exp057_structured_graph]]

## Functions

- **`_load(c: Config, ckpt: Path, device: str)`**
- **`_build_terms(model, batch, c: Config, *, epoch: int, beta: float, imp_log_std: float)`**
- **`_grad_probe(term_name: str, term: torch.Tensor, model)`**
- **`main()`**

## Imports

- [[experiments.exp057_structured_graph.codes.__init__]]
- [[experiments.exp057_structured_graph.codes.dataloader_multifreq]]
- [[experiments.exp057_structured_graph.codes.run_epoch_encode]]
- [[experiments.exp057_structured_graph.codes.train_core]]
- [[experiments.exp057_structured_graph.codes.train_vae_simple]]
- [[experiments.exp057_structured_graph.codes.training_guard]]

## External dependencies

`torch`
