---
title: train_vae_simple
type: code
path: experiments/exp050/codes/train_vae_simple.py
group: experiments/exp050/codes
experiment: exp050
loc: 299
tags: [code, exp050]
---

# train_vae_simple

> Train VAE for exp050 — pixel peak blob losses, no percentile training terms.

**Source:** `experiments/exp050/codes/train_vae_simple.py` · 299 lines
**Experiment:** [[exp050]]

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
- [[experiments.exp050.codes.__init__]]
- [[experiments.exp050.codes.dataloader_multifreq]]
- [[experiments.exp050.codes.eval_spatial_metrics]]
- [[experiments.exp050.codes.heatmap_peak_losses]]
- [[experiments.exp050.codes.run_epoch_encode]]
- [[experiments.exp050.codes.spatial_metrics]]
- [[experiments.exp050.codes.synthetic_freq_blend]]
- [[experiments.exp050.codes.vae_poe_freq]]
- [[heatmap_z_clip]]
- [[norm_stats]]
- [[repo_paths]]

## External dependencies

`repo_paths`, `src_vae`, `torch`
