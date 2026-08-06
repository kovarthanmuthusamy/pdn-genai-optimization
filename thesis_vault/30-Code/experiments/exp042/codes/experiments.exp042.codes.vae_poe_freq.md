---
title: vae_poe_freq
type: code
path: experiments/exp042/codes/vae_poe_freq.py
group: experiments/exp042/codes
experiment: exp042
loc: 222
tags: [code, exp042]
---

# vae_poe_freq

> Multi-input VAE with PI_freq as a dedicated PoE expert (exp042).

**Source:** `experiments/exp042/codes/vae_poe_freq.py` · 222 lines
**Experiment:** [[exp042]]

## Purpose

```text
Multi-input VAE with PI_freq as a dedicated PoE expert (exp042).

Adds a frequency-only expert that writes **heatmap-private** latent dims from
``PI_freq`` alone, fused via PoE with occ/imp (layout path) or hm/occ/imp (full
encode). Occ/imp experts stay K-only; PI_freq does not condition occ/imp encoders.
```

## Classes

- **`MultiInputVAEPoeFreq(MultiInputVAE)`** — PoE fusion includes layout experts + PI_freq expert on private dims.

## Imports

- [[experiments.exp038_true_multi.codes.vae_multi_input_simple]]

## Imported by

- [[exp042_eval_common]]
- [[experiments.exp042.codes.inference_vae]]
- [[experiments.exp042.codes.train_vae_simple]]

## External dependencies

`torch`
