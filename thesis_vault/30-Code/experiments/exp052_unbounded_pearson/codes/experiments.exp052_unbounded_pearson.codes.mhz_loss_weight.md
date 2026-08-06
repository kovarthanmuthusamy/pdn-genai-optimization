---
title: mhz_loss_weight
type: code
path: experiments/exp052_unbounded_pearson/codes/mhz_loss_weight.py
group: experiments/exp052_unbounded_pearson/codes
experiment: exp052_unbounded_pearson
loc: 57
tags: [code, exp052_unbounded_pearson]
---

# mhz_loss_weight

> Per-sample heatmap loss multipliers vs PI frequency (MHz).

**Source:** `experiments/exp052_unbounded_pearson/codes/mhz_loss_weight.py` · 57 lines
**Experiment:** [[exp052_unbounded_pearson]]

## Functions

- **`pi_norm_to_mhz(pi_norm: torch.Tensor)`**
- **`mhz_loss_weight(pi_norm: torch.Tensor, c: Config)`** — Per-sample multiplier: low MHz -> ``min``, high MHz -> ``max``.
- **`weighted_mean(loss_per: torch.Tensor, weight: torch.Tensor)`**

## Imports

- [[experiments.exp038_true_multi.codes.train_vae_simple]]
- [[pi_freq_utils]]

## Imported by

- [[experiments.exp052_unbounded_pearson.codes.diagnose_nan_grad]]
- [[experiments.exp052_unbounded_pearson.codes.run_epoch_encode]]
- [[experiments.exp052_unbounded_pearson.codes.train_vae_simple]]

## External dependencies

`src_vae`, `torch`
