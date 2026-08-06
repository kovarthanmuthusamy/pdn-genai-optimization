---
title: run_epoch_encode
type: code
path: experiments/exp046/codes/run_epoch_encode.py
group: experiments/exp046/codes
experiment: exp046
loc: 334
tags: [code, exp046]
---

# run_epoch_encode

> Training epoch loop for exp046 — encode z + U-Net skips + spatial losses on bounded data.

**Source:** `experiments/exp046/codes/run_epoch_encode.py` · 334 lines
**Experiment:** [[exp046]]

## Functions

- **`_latent_distill_loss(mu_student: torch.Tensor, mu_teacher: torch.Tensor, c: Config, base)`** — Pull the layout (student) latent toward the encode (teacher) latent.
- **`_forward_train_batch(model, hm_enc: torch.Tensor, occ: torch.Tensor, imp: torch.Tensor, K: torch.Tensor, pi: torch.Tensor, c: Config, *, train: bool)`** — Encode for KL; decode with teacher U-Net skips on encode batches.
- **`_cross_freq_decode(c: Config, base, *, hm_enc, occ, imp, K, pi, z_decode, pi_alt, hm_alt, ps: float, dynrange_weight: float | None)`** — Encode z + teacher skips for cross-freq heatmap loss.
- **`_run_epoch(model, loader, c: Config, epoch: int, beta: float, md: float, physics: PhysicsLoss | None, pw: tuple[float, float, float] | None, imp_log_std: float, hm_log_mean: float, hm_log_std: float, *, train: bool, optimizer=None, scaler=None, collect_per_k: bool=False)`**

## Imports

- [[experiments.exp038_true_multi.codes.train_vae_simple]]

## Imported by

- [[experiments.exp046.codes.train_vae_simple]]

## External dependencies

`torch`
