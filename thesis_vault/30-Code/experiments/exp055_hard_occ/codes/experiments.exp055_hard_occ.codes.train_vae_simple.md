---
title: train_vae_simple
type: code
path: experiments/exp055_hard_occ/codes/train_vae_simple.py
group: experiments/exp055_hard_occ/codes
experiment: exp055_hard_occ
loc: 351
tags: [code, exp055_hard_occ]
---

# train_vae_simple

> Train VAE for exp055 — tier_a + peak/valley extrema losses.

**Source:** `experiments/exp055_hard_occ/codes/train_vae_simple.py` · 351 lines
**Experiment:** [[exp055_hard_occ]]

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
- **`_vae_loss_exp055(recon_hm, recon_occ, recon_imp, tgt_hm, tgt_occ, tgt_imp, mu, logvar, beta, c, expert_stats, *, epoch, K, physics, pw, pi_freq, imp_log_std, ps, apply_k, weight_overrides=None)`**
- **`_append_heatmap_peak_split_csv(metrics_dir, epoch, tr, val)`**
- **`_apply_yaml_exp055(c: Config)`**
- **`_impedance_weights_exp055(c: Config)`**
- **`_impedance_loss_exp055(recon, target, c, imp_log_std, ps, *, epoch)`**
- **`_patch_training()`**
- **`_patch_quiet_logging()`**
- **`train_vae()`**
- **`main()`**

## Imports

- [[exp055_common]]
- [[experiments.exp055_hard_occ.codes.__init__]]
- [[experiments.exp055_hard_occ.codes.dataloader_multifreq]]
- [[experiments.exp055_hard_occ.codes.distributed_train]]
- [[experiments.exp055_hard_occ.codes.eval_off_anchor]]
- [[experiments.exp055_hard_occ.codes.heatmap_peak_losses]]
- [[experiments.exp055_hard_occ.codes.impedance_spectrum_loss]]
- [[experiments.exp055_hard_occ.codes.run_epoch_encode]]
- [[experiments.exp055_hard_occ.codes.spatial_metrics]]
- [[experiments.exp055_hard_occ.codes.train_core]]
- [[experiments.exp055_hard_occ.codes.training_guard]]
- [[experiments.exp055_hard_occ.codes.vae_poe_freq]]
- [[heatmap_z_clip]]
- [[norm_stats]]
- [[repo_paths]]

## Imported by

- [[experiments.exp055_hard_occ.codes.diagnose_nan_grad]]

## External dependencies

`builtins`, `repo_paths`, `src_vae`, `torch`
