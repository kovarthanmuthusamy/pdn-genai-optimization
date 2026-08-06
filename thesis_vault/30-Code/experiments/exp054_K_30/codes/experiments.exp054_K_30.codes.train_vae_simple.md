---
title: train_vae_simple
type: code
path: experiments/exp054_K_30/codes/train_vae_simple.py
group: experiments/exp054_K_30/codes
experiment: exp054_K_30
loc: 340
tags: [code, exp054_K_30]
---

# train_vae_simple

> Train VAE for exp054 — tier_a + peak/valley extrema losses.

**Source:** `experiments/exp054_K_30/codes/train_vae_simple.py` · 340 lines
**Experiment:** [[exp054_K_30]]

## Constants

| Name | Value |
|------|-------|
| `_REPO` | `Path(__file__).resolve().parents[3]` |
| `_EXTREMA_LOC` | `(('heatmap_peak_loc_weight', 'heatmap_peak_loc_temperature', 'peak'), ('heatmap_valley_lo…` |
| `_SCHEDULE_KEYS` | `('cross_freq_start_epoch', 'impedance_peak_start_epoch', 'impedance_peak_focus_epoch', 'b…` |

## Classes

- **`Config(_tr.Config)`**

## Functions

- **`_on_train_epoch_start(epoch_1based: int, c: Config, train_ld)`**
- **`build_vae_model(c: Config)`**
- **`_extrema_loc_losses(recon, target, c)`**
- **`_heatmap_loss_with_spatial(recon, target, c, ps, *, lite=False, **_)`**
- **`_on_stats_loaded(c: Config, raw: dict)`**
- **`_log1p_weight(c, weight_attr, focus_attr, epoch, focus_ep)`**
- **`_apply_extrema_log1p_losses(losses, recon_hm, tgt_hm, c, *, epoch, pi_freq, focus_ep)`**
- **`_vae_loss_exp054(recon_hm, recon_occ, recon_imp, tgt_hm, tgt_occ, tgt_imp, mu, logvar, beta, c, expert_stats, *, epoch, K, physics, pw, pi_freq, imp_log_std, ps, apply_k, weight_overrides=None)`**
- **`_append_heatmap_peak_split_csv(metrics_dir, epoch, tr, val)`**
- **`_apply_yaml_exp054(c: Config)`**
- **`_impedance_weights_exp054(c: Config)`**
- **`_impedance_loss_exp054(recon, target, c, imp_log_std, ps, *, epoch)`**
- **`_patch_training()`**
- **`_patch_quiet_logging()`**
- **`train_vae()`**
- **`main()`**

## Imports

- [[exp054_common]]
- [[experiments.exp054_K_30.codes.__init__]]
- [[experiments.exp054_K_30.codes.dataloader_multifreq]]
- [[experiments.exp054_K_30.codes.distributed_train]]
- [[experiments.exp054_K_30.codes.eval_off_anchor]]
- [[experiments.exp054_K_30.codes.heatmap_peak_losses]]
- [[experiments.exp054_K_30.codes.impedance_spectrum_loss]]
- [[experiments.exp054_K_30.codes.run_epoch_encode]]
- [[experiments.exp054_K_30.codes.spatial_metrics]]
- [[experiments.exp054_K_30.codes.train_core]]
- [[experiments.exp054_K_30.codes.training_guard]]
- [[experiments.exp054_K_30.codes.vae_poe_freq]]
- [[heatmap_z_clip]]
- [[norm_stats]]
- [[repo_paths]]

## Imported by

- [[experiments.exp054_K_30.codes.diagnose_nan_grad]]

## External dependencies

`builtins`, `repo_paths`, `src_vae`, `torch`
