---
title: run_epoch_encode
type: code
path: experiments/exp057_structured_graph/codes/run_epoch_encode.py
group: experiments/exp057_structured_graph/codes
experiment: exp057_structured_graph
loc: 236
tags: [code, exp057_structured_graph]
---

# run_epoch_encode

> exp055 training epoch — layout path + log1p peak/valley.

**Source:** `experiments/exp057_structured_graph/codes/run_epoch_encode.py` · 236 lines
**Experiment:** [[exp057_structured_graph]]

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

- [[experiments.exp057_structured_graph.codes.distributed_train]]
- [[experiments.exp057_structured_graph.codes.train_core]]
- [[experiments.exp057_structured_graph.codes.training_guard]]

## Imported by

- [[experiments.exp057_structured_graph.codes.diagnose_nan_grad]]
- [[experiments.exp057_structured_graph.codes.train_vae_simple]]

## External dependencies

`torch`
