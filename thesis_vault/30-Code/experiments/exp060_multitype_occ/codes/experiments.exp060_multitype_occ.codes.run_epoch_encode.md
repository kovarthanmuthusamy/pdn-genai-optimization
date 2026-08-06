---
title: run_epoch_encode
type: code
path: experiments/exp060_multitype_occ/codes/run_epoch_encode.py
group: experiments/exp060_multitype_occ/codes
experiment: exp060_multitype_occ
loc: 257
tags: [code, exp060_multitype_occ, uncommitted]
---

# run_epoch_encode

> exp055 training epoch — layout path + log1p peak/valley.

**Source:** `experiments/exp060_multitype_occ/codes/run_epoch_encode.py` · 257 lines
**Git:** uncommitted — not yet tracked
**Experiment:** [[exp060_multitype_occ]]

## Constants

| Name | Value |
|------|-------|
| `_HM_KEYS` | `('heatmap_loss', 'heatmap_loss_tier_a', 'heatmap_peak_log1p_loss', 'heatmap_valley_log1p_…` |

## Functions

- **`_use_amp(c: Config, *, train: bool)`**
- **`_latent_distill_loss(mu_s, mu_t, c, base, lv_s=None, lv_t=None)`**
- **`_forward_train_batch(model, hm_enc, occ, imp, K, pi, c, *, train: bool)`**
- **`_cross_freq_decode(c, base, *, hm_enc, occ, imp, K, pi, z_decode, pi_alt, hm_alt, ps, z_encode=None)`**
- **`_run_epoch(model, loader, c: Config, epoch, beta, md, physics, pw, imp_log_std, hm_log_mean, hm_log_std, *, train: bool, optimizer=None, scaler=None, collect_per_k: bool=False)`**

## Imports

- [[experiments.exp060_multitype_occ.codes.distributed_train]]
- [[experiments.exp060_multitype_occ.codes.spatial_metrics]]
- [[experiments.exp060_multitype_occ.codes.train_core]]
- [[experiments.exp060_multitype_occ.codes.training_guard]]

## Imported by

- [[experiments.exp060_multitype_occ.codes.train_vae_simple]]

## External dependencies

`torch`
