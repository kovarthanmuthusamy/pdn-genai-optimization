---
title: train_vae_simple
type: code
path: experiments/exp051_new_datas_appended/codes/train_vae_simple.py
group: experiments/exp051_new_datas_appended/codes
experiment: exp051_new_datas_appended
loc: 422
tags: [code, exp051_new_datas_appended]
---

# train_vae_simple

> Train VAE for exp051 — pixel peak blob losses, no percentile training terms.

**Source:** `experiments/exp051_new_datas_appended/codes/train_vae_simple.py` · 422 lines
**Experiment:** [[exp051_new_datas_appended]]

## Constants

| Name | Value |
|------|-------|
| `_REPO_BOOT` | `Path(__file__).resolve().parents[3]` |
| `_IMPEDANCE_TRAIN_TAGS` | `('impedance_encoder', 'impedance_mu', 'impedance_logvar', 'impedance_decoder', 'imp_fc')` |

## Classes

- **`Config(_tr.Config)`**

## Functions

- **`_on_train_epoch_start(epoch_1based: int, c: Config, train_ld)`**
- **`build_vae_model(c: Config)`**
- **`_heatmap_loss_with_spatial(recon: torch.Tensor, target: torch.Tensor, c: _tr.Config, ps: float, *, dynrange_weight: float | None=None, lite: bool=False)`**
- **`_prepare_batch(batch: dict, c: _tr.Config)`**
- **`_phase_weights_with_freeze(epoch: int, c: Config)`**
- **`_unfreeze_impedance_modules(c: Config)`** — Unfreeze impedance path; keep occupancy decoder frozen.
- **`_on_after_checkpoint_load(c: Config, model, start: int)`**
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
- [[experiments.exp051_new_datas_appended.codes.__init__]]
- [[experiments.exp051_new_datas_appended.codes.dataloader_multifreq]]
- [[experiments.exp051_new_datas_appended.codes.eval_spatial_metrics]]
- [[experiments.exp051_new_datas_appended.codes.heatmap_peak_losses]]
- [[experiments.exp051_new_datas_appended.codes.run_epoch_encode]]
- [[experiments.exp051_new_datas_appended.codes.sampler_curriculum]]
- [[experiments.exp051_new_datas_appended.codes.spatial_metrics]]
- [[experiments.exp051_new_datas_appended.codes.synthetic_freq_blend]]
- [[experiments.exp051_new_datas_appended.codes.vae_poe_freq]]
- [[heatmap_z_clip]]
- [[norm_stats]]
- [[repo_paths]]

## External dependencies

`repo_paths`, `src_vae`, `torch`
