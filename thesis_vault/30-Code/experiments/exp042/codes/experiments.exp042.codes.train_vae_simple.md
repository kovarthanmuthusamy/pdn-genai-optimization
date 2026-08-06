---
title: train_vae_simple
type: code
path: experiments/exp042/codes/train_vae_simple.py
group: experiments/exp042/codes
experiment: exp042
loc: 108
tags: [code, exp042]
---

# train_vae_simple

> Train VAE for exp042 — PI_freq PoE expert on heatmap-private dims.

**Source:** `experiments/exp042/codes/train_vae_simple.py` · 108 lines
**Experiment:** [[exp042]]

## Purpose

```text
Train VAE for exp042 — PI_freq PoE expert on heatmap-private dims.

Same trainer as exp038/exp041 with ``MultiInputVAEPoeFreq`` and synthetic blends.
```

## Constants

| Name | Value |
|------|-------|
| `_REPO_BOOT` | `Path(__file__).resolve().parents[3]` |

## Classes

- **`Config(_tr.Config)`**

## Functions

- **`build_vae_model(c: Config)`**
- **`_prepare_batch(batch: dict, c: _tr.Config)`**
- **`_patch_training()`**
- **`_print_layout_sweep_training(c: _tr.Config)`** — Confirm layout-path + cross-freq + freq PoE settings (sweep-aligned).
- **`train_vae()`**
- **`main()`**

## Imports

- [[experiments.exp038_true_multi.codes.train_vae_simple]]
- [[experiments.exp042.codes.synthetic_freq_blend]]
- [[experiments.exp042.codes.vae_poe_freq]]
- [[repo_paths]]

## External dependencies

`repo_paths`
