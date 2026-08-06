---
title: run_epoch_encode
type: code
path: experiments/exp045/codes/run_epoch_encode.py
group: experiments/exp045/codes
experiment: exp045
loc: 287
tags: [code, exp045]
---

# run_epoch_encode

> Training epoch loop for exp045 — encode z + U-Net skips on cross-freq decode.

**Source:** `experiments/exp045/codes/run_epoch_encode.py` · 287 lines
**Experiment:** [[exp045]]

## Functions

- **`_forward_train_batch(model, hm_enc: torch.Tensor, occ: torch.Tensor, imp: torch.Tensor, K: torch.Tensor, pi: torch.Tensor, c: Config, *, train: bool)`** — Encode for KL; decode with teacher U-Net skips on encode batches.
- **`_cross_freq_decode(c: Config, base, *, hm_enc, occ, imp, K, pi, z_decode, pi_alt, hm_alt, ps: float, dynrange_weight: float | None)`** — Encode z + teacher skips for cross-freq heatmap loss.
- **`_run_epoch(model, loader, c: Config, epoch: int, beta: float, md: float, physics: PhysicsLoss | None, pw: tuple[float, float, float] | None, imp_log_std: float, hm_log_mean: float, hm_log_std: float, *, train: bool, optimizer=None, scaler=None, collect_per_k: bool=False)`**

## Imports

- [[experiments.exp038_true_multi.codes.train_vae_simple]]

## Imported by

- [[experiments.exp045.codes.train_vae_simple]]

## External dependencies

`torch`
