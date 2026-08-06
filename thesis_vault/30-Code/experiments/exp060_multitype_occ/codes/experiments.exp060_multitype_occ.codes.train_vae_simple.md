---
title: train_vae_simple
type: code
path: experiments/exp060_multitype_occ/codes/train_vae_simple.py
group: experiments/exp060_multitype_occ/codes
experiment: exp060_multitype_occ
loc: 478
tags: [code, exp060_multitype_occ, uncommitted]
---

# train_vae_simple

> Train VAE for exp060 Graph VAE — tier_a + peak/valley extrema losses.

**Source:** `experiments/exp060_multitype_occ/codes/train_vae_simple.py` · 478 lines
**Git:** uncommitted — not yet tracked
**Experiment:** [[exp060_multitype_occ]]

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
- **`_apply_peak_loc_losses(losses, recon_hm, tgt_hm, c)`** — Soft-argmax peak (and optional valley) coordinate MSE — peak-position aware.
- **`_heatmap_spectral_loss(recon, target, c, *, pi_freq=None)`** — 2D-FFT magnitude L1 between recon and target heatmaps (anti-blur).
- **`_vae_loss_exp055(recon_hm, recon_occ, recon_imp, tgt_hm, tgt_occ, tgt_imp, mu, logvar, beta, c, expert_stats, *, epoch, K, physics, pw, pi_freq, imp_log_std, ps, apply_k, weight_overrides=None)`**
- **`_append_heatmap_peak_split_csv(metrics_dir, epoch, tr, val)`**
- **`_apply_yaml_exp055(c: Config)`** — Load config.yaml then optional ``VAE_CONFIG_PATH`` (AL fine-tune runtime).
- **`_impedance_weights_exp055(c: Config)`**
- **`_impedance_loss_exp055(recon, target, c, imp_log_std, ps, *, epoch)`**
- **`_patch_training()`**
- **`_patch_quiet_logging()`**
- **`train_vae()`**
- **`main()`**

## Imports

- [[exp060_common]]
- [[experiments.exp060_multitype_occ.codes.__init__]]
- [[experiments.exp060_multitype_occ.codes.dataloader_multifreq]]
- [[experiments.exp060_multitype_occ.codes.distributed_train]]
- [[experiments.exp060_multitype_occ.codes.eval_off_anchor]]
- [[experiments.exp060_multitype_occ.codes.heatmap_peak_losses]]
- [[experiments.exp060_multitype_occ.codes.impedance_spectrum_loss]]
- [[experiments.exp060_multitype_occ.codes.run_epoch_encode]]
- [[experiments.exp060_multitype_occ.codes.spatial_metrics]]
- [[experiments.exp060_multitype_occ.codes.train_core]]
- [[experiments.exp060_multitype_occ.codes.training_guard]]
- [[experiments.exp060_multitype_occ.codes.vae_poe_freq]]
- [[heatmap_z_clip]]
- [[norm_stats]]
- [[pi_freq_utils]]
- [[repo_paths]]

## External dependencies

`builtins`, `repo_paths`, `src_vae`, `torch`
