---
title: train_vae_simple
type: code
path: experiments/exp045/codes/train_vae_simple.py
group: experiments/exp045/codes
experiment: exp045
loc: 220
tags: [code, exp045]
---

# train_vae_simple

> Train VAE for exp045 — unbounded norm + U-Net skips + encode-first + robust losses.

**Source:** `experiments/exp045/codes/train_vae_simple.py` · 220 lines
**Experiment:** [[exp045]]

## Constants

| Name | Value |
|------|-------|
| `_REPO_BOOT` | `Path(__file__).resolve().parents[3]` |

## Classes

- **`Config(_tr.Config)`**

## Functions

- **`build_vae_model(c: Config)`**
- **`_on_stats_loaded(c: Config, raw: dict)`**
- **`_prepare_batch(batch: dict, c: _tr.Config)`**
- **`_patch_training()`**
- **`_patch_spatial_eval()`**
- **`_print_training_summary(c: _tr.Config)`**
- **`train_vae()`**
- **`main()`**

## Imports

- [[experiments.exp038_true_multi.codes.eval_cross_freq]]
- [[experiments.exp038_true_multi.codes.train_vae_simple]]
- [[experiments.exp045.codes.eval_spatial_metrics]]
- [[experiments.exp045.codes.run_epoch_encode]]
- [[experiments.exp045.codes.synthetic_freq_blend]]
- [[experiments.exp045.codes.vae_poe_freq]]
- [[repo_paths]]
- [[unbounded_heatmap_loss]]

## External dependencies

`repo_paths`, `torch`
