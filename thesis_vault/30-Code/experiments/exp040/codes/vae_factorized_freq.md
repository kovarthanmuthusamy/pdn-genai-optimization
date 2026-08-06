---
title: vae_factorized_freq
type: code
path: experiments/exp040/codes/vae_factorized_freq.py
group: experiments/exp040/codes
experiment: exp040
loc: 205
tags: [code, exp040]
---

# vae_factorized_freq

> Factorized-frequency heatmap VAE (exp040).

**Source:** `experiments/exp040/codes/vae_factorized_freq.py` · 205 lines
**Experiment:** [[exp040]]

## Purpose

```text
Factorized-frequency heatmap VAE (exp040).

Heatmap decode:
    H(x, y | z, K, MHz) = sum_k alpha_k(MHz) * B_k(x, y | z, K)  [+ optional residual]

- Spatial modes B_k depend on layout latent z and K only (not PI_freq).
- Mixing weights alpha_k depend on PI_freq only (shared across layouts).
- Occupancy / impedance paths unchanged from MultiInputVAE (K-only experts).
```

## Classes

- **`FactorizedFreqVAE(MultiInputVAE)`** — Multifreq VAE with PI-frequency factorized heatmap decoder.

## Functions

- **`modes_orthogonality_loss(bases: torch.Tensor)`** — Penalize correlated spatial modes (B, M, H, W).

## Imports

- [[experiments.exp038_true_multi.codes.freq_conditioning]]
- [[experiments.exp038_true_multi.codes.freq_inference_utils]]
- [[experiments.exp038_true_multi.codes.vae_multi_input_simple]]
- [[pi_freq_utils]]

## Imported by

- [[experiments.exp040.codes.inference_vae]]
- [[experiments.exp040.codes.train_vae_simple]]

## External dependencies

`src_vae`, `torch`
