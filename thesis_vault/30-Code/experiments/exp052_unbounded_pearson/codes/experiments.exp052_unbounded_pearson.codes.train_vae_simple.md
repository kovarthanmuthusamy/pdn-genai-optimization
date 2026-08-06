---
title: train_vae_simple
type: code
path: experiments/exp052_unbounded_pearson/codes/train_vae_simple.py
group: experiments/exp052_unbounded_pearson/codes
experiment: exp052_unbounded_pearson
loc: 493
tags: [code, exp052_unbounded_pearson]
---

# train_vae_simple

> Train VAE for exp052 — Pearson + grad + phys top-k blob (no percentile/dynrange/z-clip).

**Source:** `experiments/exp052_unbounded_pearson/codes/train_vae_simple.py` · 493 lines
**Experiment:** [[exp052_unbounded_pearson]]

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
- **`_heatmap_loss_with_spatial(recon: torch.Tensor, target: torch.Tensor, c: _tr.Config, ps: float, *, lite: bool=False, **_)`**
- **`_prepare_batch(batch: dict, c: _tr.Config)`**
- **`_phase_weights_with_freeze(epoch: int, c: Config)`**
- **`_unfreeze_impedance_modules(c: Config)`** — Unfreeze impedance path; keep occupancy decoder frozen.
- **`_on_after_checkpoint_load(c: Config, model, start: int)`**
- **`_freeze_occ_imp_decoders(c: Config)`**
- **`_on_stats_loaded(c: Config, raw: dict)`** — Attach unbounded norm bundle (exp052 requires unbounded stats).
- **`_vae_loss_mhz_weighted(recon_hm, recon_occ, recon_imp, tgt_hm, tgt_occ, tgt_imp, mu, logvar, beta, c, expert_stats, *, epoch, K, physics, pw, pi_freq, imp_log_std, ps, apply_k, weight_overrides=None)`**
- **`_patch_training()`**
- **`_patch_dataloader()`**
- **`_patch_spatial_eval()`**
- **`_print_training_summary(c: _tr.Config)`**
- **`train_vae()`**
- **`main()`**

## Imports

- [[experiments.exp038_true_multi.codes.eval_cross_freq]]
- [[experiments.exp038_true_multi.codes.train_vae_simple]]
- [[experiments.exp052_unbounded_pearson.codes.__init__]]
- [[experiments.exp052_unbounded_pearson.codes.dataloader_multifreq]]
- [[experiments.exp052_unbounded_pearson.codes.eval_spatial_metrics]]
- [[experiments.exp052_unbounded_pearson.codes.heatmap_peak_losses]]
- [[experiments.exp052_unbounded_pearson.codes.mhz_loss_weight]]
- [[experiments.exp052_unbounded_pearson.codes.run_epoch_encode]]
- [[experiments.exp052_unbounded_pearson.codes.sampler_curriculum]]
- [[experiments.exp052_unbounded_pearson.codes.spatial_metrics]]
- [[experiments.exp052_unbounded_pearson.codes.synthetic_freq_blend]]
- [[experiments.exp052_unbounded_pearson.codes.training_guard]]
- [[experiments.exp052_unbounded_pearson.codes.vae_poe_freq]]
- [[norm_stats]]
- [[repo_paths]]

## Imported by

- [[experiments.exp052_unbounded_pearson.codes.diagnose_nan_grad]]

## External dependencies

`repo_paths`, `src_vae`, `torch`
