---
title: run_epoch_encode
type: code
path: experiments/exp044/codes/run_epoch_encode.py
group: experiments/exp044/codes
experiment: exp044
loc: 246
tags: [code, exp044]
---

# run_epoch_encode

> Training epoch loop for exp044 — cross-freq always uses encode z (GT heatmap in latent).

**Source:** `experiments/exp044/codes/run_epoch_encode.py` · 246 lines
**Experiment:** [[exp044]]

## Functions

- **`_cross_freq_z(c: Config, base, *, hm_enc, occ, imp, K, pi, z_decode)`** — Cross-freq decode z: encode path (heatmap in latent) unless layout-only override.
- **`_run_epoch(model, loader, c: Config, epoch: int, beta: float, md: float, physics: PhysicsLoss | None, pw: tuple[float, float, float] | None, imp_log_std: float, hm_log_mean: float, hm_log_std: float, *, train: bool, optimizer=None, scaler=None, collect_per_k: bool=False)`**

## Imports

- [[experiments.exp038_true_multi.codes.train_vae_simple]]

## Imported by

- [[experiments.exp044.codes.train_vae_simple]]

## External dependencies

`torch`
