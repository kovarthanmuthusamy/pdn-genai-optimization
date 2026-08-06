---
title: run_epoch_encode
type: code
path: experiments/exp056_graph_vae/codes/run_epoch_encode.py
group: experiments/exp056_graph_vae/codes
experiment: exp056_graph_vae
loc: 223
tags: [code, exp056_graph_vae]
---

# run_epoch_encode

> exp055 training epoch — layout path + log1p peak/valley.

**Source:** `experiments/exp056_graph_vae/codes/run_epoch_encode.py` · 223 lines
**Experiment:** [[exp056_graph_vae]]

## Constants

| Name | Value |
|------|-------|
| `_HM_KEYS` | `('heatmap_loss', 'heatmap_loss_tier_a', 'heatmap_peak_log1p_loss', 'heatmap_valley_log1p_…` |

## Functions

- **`_use_amp(c: Config, *, train: bool)`**
- **`_latent_distill_loss(mu_s, mu_t, c, base, lv_s=None, lv_t=None)`**
- **`_forward_train_batch(model, hm_enc, occ, imp, K, pi, c, *, train: bool)`**
- **`_cross_freq_decode(c, base, *, hm_enc, occ, imp, K, pi, z_decode, pi_alt, hm_alt, ps, z_encode=None, encode_skips_cache=None)`**
- **`_run_epoch(model, loader, c: Config, epoch, beta, md, physics, pw, imp_log_std, hm_log_mean, hm_log_std, *, train: bool, optimizer=None, scaler=None, collect_per_k: bool=False)`**

## Imports

- [[experiments.exp056_graph_vae.codes.distributed_train]]
- [[experiments.exp056_graph_vae.codes.train_core]]
- [[experiments.exp056_graph_vae.codes.training_guard]]

## Imported by

- [[experiments.exp056_graph_vae.codes.diagnose_nan_grad]]
- [[experiments.exp056_graph_vae.codes.train_vae_simple]]

## External dependencies

`torch`
