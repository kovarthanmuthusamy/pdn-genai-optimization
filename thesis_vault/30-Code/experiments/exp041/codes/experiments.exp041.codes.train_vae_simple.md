---
title: train_vae_simple
type: code
path: experiments/exp041/codes/train_vae_simple.py
group: experiments/exp041/codes
experiment: exp041
loc: 73
tags: [code, exp041]
---

# train_vae_simple

> Train VAE for exp041 (inverse-K subsampled multifreq dataset).

**Source:** `experiments/exp041/codes/train_vae_simple.py` · 73 lines
**Experiment:** [[exp041]]

## Purpose

```text
Train VAE for exp041 (inverse-K subsampled multifreq dataset).

Uses ``experiments/exp038_true_multi/codes/train_vae_simple.py`` with overrides from
``config.yaml`` (VAE_EXPERIMENT_DIR). Adds optional synthetic mid-MHz heatmap blends.
```

## Constants

| Name | Value |
|------|-------|
| `_REPO_BOOT` | `Path(__file__).resolve().parents[3]` |

## Classes

- **`Config(_tr.Config)`**

## Functions

- **`_prepare_batch(batch: dict, c: _tr.Config)`**
- **`_patch_training()`**
- **`train_vae()`**
- **`main()`**

## Imports

- [[experiments.exp038_true_multi.codes.train_vae_simple]]
- [[experiments.exp041.codes.synthetic_freq_blend]]
- [[repo_paths]]

## External dependencies

`repo_paths`
