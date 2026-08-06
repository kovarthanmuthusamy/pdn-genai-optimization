---
title: train_vae_simple
type: code
path: experiments/exp049/codes/train_vae_simple.py
group: experiments/exp049/codes
experiment: exp049
loc: 291
tags: [code, exp049]
---

# train_vae_simple

> Train VAE for exp049 — high-freq layout improvements, no early stop.

**Source:** `experiments/exp049/codes/train_vae_simple.py` · 291 lines
**Experiment:** [[exp049]]

## Constants

| Name | Value |
|------|-------|
| `_REPO_BOOT` | `Path(__file__).resolve().parents[3]` |

## Classes

- **`Config(_tr.Config)`**

## Functions

- **`build_vae_model(c: Config)`**
- **`_heatmap_loss_with_spatial(recon: torch.Tensor, target: torch.Tensor, c: _tr.Config, ps: float, *, dynrange_weight: float | None=None, lite: bool=False)`**
- **`_prepare_batch(batch: dict, c: _tr.Config)`**
- **`_phase_weights_with_freeze(epoch: int, c: Config)`**
- **`_freeze_occ_imp_decoders(c: Config)`**
- **`_on_stats_loaded(c: Config, raw: dict)`** — Attach norm bundle; disable z-clip for unbounded robust stats.
- **`_patch_training()`**
- **`_patch_dataloader()`**
- **`_patch_spatial_eval()`**
- **`_print_training_summary(c: _tr.Config)`**
- **`train_vae()`**
- **`main()`**

## Imports

- [[experiments.exp038_true_multi.codes.eval_cross_freq]]
- [[experiments.exp038_true_multi.codes.train_vae_simple]]
- [[experiments.exp049.codes.__init__]]
- [[experiments.exp049.codes.dataloader_multifreq]]
- [[experiments.exp049.codes.eval_spatial_metrics]]
- [[experiments.exp049.codes.run_epoch_encode]]
- [[experiments.exp049.codes.spatial_metrics]]
- [[experiments.exp049.codes.synthetic_freq_blend]]
- [[experiments.exp049.codes.vae_poe_freq]]
- [[heatmap_z_clip]]
- [[norm_stats]]
- [[repo_paths]]

## External dependencies

`repo_paths`, `src_vae`, `torch`
