---
title: train_vae_simple
type: code
path: experiments/exp044/codes/train_vae_simple.py
group: experiments/exp044/codes
experiment: exp044
loc: 116
tags: [code, exp044]
---

# train_vae_simple

> Train VAE for exp044 — exp042 freq PoE + encode-first latent (latent-opt path).

**Source:** `experiments/exp044/codes/train_vae_simple.py` · 116 lines
**Experiment:** [[exp044]]

## Purpose

```text
Train VAE for exp044 — exp042 freq PoE + encode-first latent (latent-opt path).

Same dataset/norm as exp042 (``data_multifreq_norm_z_score`` log1p z-score). Training emphasizes
Path A: ``z = PoE(heatmap_encoder(GT), occ, imp, freq)`` for decode + cross-freq loss.
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
- **`_print_encode_first_training(c: _tr.Config)`**
- **`train_vae()`**
- **`main()`**

## Imports

- [[experiments.exp038_true_multi.codes.train_vae_simple]]
- [[experiments.exp044.codes.run_epoch_encode]]
- [[experiments.exp044.codes.synthetic_freq_blend]]
- [[experiments.exp044.codes.vae_poe_freq]]
- [[repo_paths]]

## External dependencies

`repo_paths`
