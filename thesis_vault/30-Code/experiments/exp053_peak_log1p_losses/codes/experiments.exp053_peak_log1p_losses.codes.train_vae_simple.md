---
title: train_vae_simple
type: code
path: experiments/exp053_peak_log1p_losses/codes/train_vae_simple.py
group: experiments/exp053_peak_log1p_losses/codes
experiment: exp053_peak_log1p_losses
loc: 579
tags: [code, exp053_peak_log1p_losses]
---

# train_vae_simple

> Train VAE for exp053 — Pearson + grad + log1p peak stack + target-only phys blob + peak boost (resume 450→600 training with log1p peak stack).

**Source:** `experiments/exp053_peak_log1p_losses/codes/train_vae_simple.py` · 579 lines
**Experiment:** [[exp053_peak_log1p_losses]]

## Constants

| Name | Value |
|------|-------|
| `_REPO_BOOT` | `Path(__file__).resolve().parents[3]` |
| `_IMPEDANCE_TRAIN_TAGS` | `('impedance_encoder', 'impedance_mu', 'impedance_logvar', 'impedance_decoder', 'imp_fc')` |

## Classes

- **`Config(_tr.Config)`**

## Functions

- **`_print_hm_breakdown(epoch_1based: int)`**
- **`_on_train_epoch_start(epoch_1based: int, c: Config, train_ld)`**
- **`build_vae_model(c: Config)`**
- **`_heatmap_loss_with_spatial(recon: torch.Tensor, target: torch.Tensor, c: _tr.Config, ps: float, *, lite: bool=False, **_)`**
- **`_prepare_batch(batch: dict, c: _tr.Config)`**
- **`_phase_weights_with_freeze(epoch: int, c: Config)`**
- **`_unfreeze_impedance_modules(c: Config)`** — Unfreeze impedance path; keep occupancy decoder frozen.
- **`_on_after_checkpoint_load(c: Config, model, start: int)`**
- **`_freeze_occ_imp_decoders(c: Config)`**
- **`_on_stats_loaded(c: Config, raw: dict)`** — Attach unbounded norm bundle (exp053 requires unbounded stats).
- **`_vae_loss_mhz_weighted(recon_hm, recon_occ, recon_imp, tgt_hm, tgt_occ, tgt_imp, mu, logvar, beta, c, expert_stats, *, epoch, K, physics, pw, pi_freq, imp_log_std, ps, apply_k, weight_overrides=None)`**
- **`_append_heatmap_peak_split_csv(metrics_dir: Path, epoch: int, tr: dict, val: dict | None)`**
- **`_patch_training()`**
- **`_patch_dataloader()`**
- **`_patch_spatial_eval()`**
- **`_print_training_summary(c: _tr.Config)`**
- **`train_vae()`**
- **`main()`**

## Imports

- [[experiments.exp038_true_multi.codes.eval_cross_freq]]
- [[experiments.exp038_true_multi.codes.train_vae_simple]]
- [[experiments.exp053_peak_log1p_losses.codes.__init__]]
- [[experiments.exp053_peak_log1p_losses.codes.dataloader_multifreq]]
- [[experiments.exp053_peak_log1p_losses.codes.eval_spatial_metrics]]
- [[experiments.exp053_peak_log1p_losses.codes.heatmap_peak_losses]]
- [[experiments.exp053_peak_log1p_losses.codes.mhz_loss_weight]]
- [[experiments.exp053_peak_log1p_losses.codes.run_epoch_encode]]
- [[experiments.exp053_peak_log1p_losses.codes.sampler_curriculum]]
- [[experiments.exp053_peak_log1p_losses.codes.spatial_metrics]]
- [[experiments.exp053_peak_log1p_losses.codes.synthetic_freq_blend]]
- [[experiments.exp053_peak_log1p_losses.codes.training_guard]]
- [[experiments.exp053_peak_log1p_losses.codes.vae_poe_freq]]
- [[norm_stats]]
- [[repo_paths]]

## Imported by

- [[experiments.exp053_peak_log1p_losses.codes.diagnose_nan_grad]]

## External dependencies

`repo_paths`, `src_vae`, `torch`
