---
title: train_vae_simple
type: code
path: experiments/exp047/codes/train_vae_simple.py
group: experiments/exp047/codes
experiment: exp047
loc: 264
tags: [code, exp047]
---

# train_vae_simple

> Train VAE for exp047 — high-freq layout improvements, no early stop.

**Source:** `experiments/exp047/codes/train_vae_simple.py` · 264 lines
**Experiment:** [[exp047]]

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
- **`_patch_training()`**
- **`_patch_dataloader()`**
- **`_patch_spatial_eval()`**
- **`_print_training_summary(c: _tr.Config)`**
- **`train_vae()`**
- **`main()`**

## Imports

- [[experiments.exp038_true_multi.codes.eval_cross_freq]]
- [[experiments.exp038_true_multi.codes.train_vae_simple]]
- [[experiments.exp047.codes.__init__]]
- [[experiments.exp047.codes.dataloader_multifreq]]
- [[experiments.exp047.codes.eval_spatial_metrics]]
- [[experiments.exp047.codes.run_epoch_encode]]
- [[experiments.exp047.codes.spatial_metrics]]
- [[experiments.exp047.codes.synthetic_freq_blend]]
- [[experiments.exp047.codes.vae_poe_freq]]
- [[repo_paths]]

## External dependencies

`repo_paths`, `torch`
