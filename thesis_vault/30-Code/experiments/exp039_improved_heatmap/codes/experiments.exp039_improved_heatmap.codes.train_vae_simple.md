---
title: train_vae_simple
type: code
path: experiments/exp039_improved_heatmap/codes/train_vae_simple.py
group: experiments/exp039_improved_heatmap/codes
experiment: exp039_improved_heatmap
loc: 60
tags: [code, exp039_improved_heatmap]
---

# train_vae_simple

> Train VAE for exp039_improved_heatmap.

**Source:** `experiments/exp039_improved_heatmap/codes/train_vae_simple.py` · 60 lines
**Experiment:** [[exp039_improved_heatmap]]

## Purpose

```text
Train VAE for exp039_improved_heatmap.

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
- [[experiments.exp039_improved_heatmap.codes.synthetic_freq_blend]]
- [[repo_paths]]

## External dependencies

`repo_paths`
