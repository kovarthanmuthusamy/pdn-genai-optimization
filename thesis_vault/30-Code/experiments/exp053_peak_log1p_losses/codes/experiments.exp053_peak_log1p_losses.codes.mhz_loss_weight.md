---
title: mhz_loss_weight
type: code
path: experiments/exp053_peak_log1p_losses/codes/mhz_loss_weight.py
group: experiments/exp053_peak_log1p_losses/codes
experiment: exp053_peak_log1p_losses
loc: 57
tags: [code, exp053_peak_log1p_losses]
---

# mhz_loss_weight

> Per-sample heatmap loss multipliers vs PI frequency (MHz).

**Source:** `experiments/exp053_peak_log1p_losses/codes/mhz_loss_weight.py` · 57 lines
**Experiment:** [[exp053_peak_log1p_losses]]

## Functions

- **`pi_norm_to_mhz(pi_norm: torch.Tensor)`**
- **`mhz_loss_weight(pi_norm: torch.Tensor, c: Config)`** — Per-sample multiplier: low MHz -> ``min``, high MHz -> ``max``.
- **`weighted_mean(loss_per: torch.Tensor, weight: torch.Tensor)`**

## Imports

- [[experiments.exp038_true_multi.codes.train_vae_simple]]
- [[pi_freq_utils]]

## Imported by

- [[experiments.exp053_peak_log1p_losses.codes.diagnose_nan_grad]]
- [[experiments.exp053_peak_log1p_losses.codes.run_epoch_encode]]
- [[experiments.exp053_peak_log1p_losses.codes.train_vae_simple]]

## External dependencies

`src_vae`, `torch`
