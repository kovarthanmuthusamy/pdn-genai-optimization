---
title: train_vae_simple
type: code
path: experiments/exp040/codes/train_vae_simple.py
group: experiments/exp040/codes
experiment: exp040
loc: 166
tags: [code, exp040]
---

# train_vae_simple

> Train exp040 FactorizedFreqVAE.

**Source:** `experiments/exp040/codes/train_vae_simple.py` · 166 lines
**Experiment:** [[exp040]]

## Purpose

```text
Train exp040 FactorizedFreqVAE.

Patches exp038 training with factorized heatmap decode, synthetic mid-MHz
blends, and optional mode-orthogonality loss.
```

## Constants

| Name | Value |
|------|-------|
| `_REPO_BOOT` | `Path(__file__).resolve().parents[3]` |
| `_RESIDUAL_PREFIXES` | `('heatmap_fc', 'heatmap_dec_deconv1', 'heatmap_dec_attn', 'heatmap_dec_film', 'heatmap_de…` |

## Classes

- **`Config(_tr.Config)`**

## Functions

- **`build_factorized_vae(c: Config)`**
- **`configure_factorized_training(model: FactorizedFreqVAE, c: Config)`** — Apply yaml flags and freeze legacy residual decoder when disabled.
- **`_prepare_batch(batch: dict, c: _tr.Config)`**
- **`vae_loss(*args, **kwargs)`**
- **`_patch_training_module()`**
- **`train_vae()`**
- **`main()`**

## Imports

- [[experiments.exp038_true_multi.codes.train_vae_simple]]
- [[experiments.exp040.codes.__init__]]
- [[experiments.exp040.codes.synthetic_freq_blend]]
- [[repo_paths]]
- [[vae_factorized_freq]]

## External dependencies

`repo_paths`, `torch`
