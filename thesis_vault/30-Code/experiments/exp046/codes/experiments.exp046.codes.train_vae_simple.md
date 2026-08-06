---
title: train_vae_simple
type: code
path: experiments/exp046/codes/train_vae_simple.py
group: experiments/exp046/codes
experiment: exp046
loc: 334
tags: [code, exp046]
---

# train_vae_simple

> Train VAE for exp046 — bounded data + spatial losses + U-Net skips + occ tower.

**Source:** `experiments/exp046/codes/train_vae_simple.py` · 334 lines
**Experiment:** [[exp046]]

## Constants

| Name | Value |
|------|-------|
| `_REPO_BOOT` | `Path(__file__).resolve().parents[3]` |

## Classes

- **`Config(_tr.Config)`**

## Functions

- **`build_vae_model(c: Config)`**
- **`_heatmap_loss_with_spatial(recon: torch.Tensor, target: torch.Tensor, c: _tr.Config, ps: float, *, dynrange_weight: float | None=None, lite: bool=False)`** — Standard bounded heatmap loss + optional Pearson + peak-loc.
- **`_prepare_batch(batch: dict, c: _tr.Config)`**
- **`_phase_weights_with_freeze(epoch: int, c: Config)`**
- **`_freeze_occ_imp_decoders(c: Config)`** — Freeze occupancy and impedance decoder parameters to reduce gradient competition.
- **`_patch_training()`**
- **`_patch_spatial_eval()`**
- **`_print_training_summary(c: _tr.Config)`**
- **`_restore_early_stop_on_resume(c: Config)`** — Restore early-stop baseline from best off-anchor eval CSV when resuming.
- **`train_vae()`**
- **`main()`**

## Imports

- [[experiments.exp038_true_multi.codes.eval_cross_freq]]
- [[experiments.exp038_true_multi.codes.train_vae_simple]]
- [[experiments.exp046.codes.eval_spatial_metrics]]
- [[experiments.exp046.codes.run_epoch_encode]]
- [[experiments.exp046.codes.spatial_metrics]]
- [[experiments.exp046.codes.synthetic_freq_blend]]
- [[experiments.exp046.codes.vae_poe_freq]]
- [[repo_paths]]

## External dependencies

`repo_paths`, `torch`
