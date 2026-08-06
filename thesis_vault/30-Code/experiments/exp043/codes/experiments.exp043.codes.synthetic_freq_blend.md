---
title: synthetic_freq_blend
type: code
path: experiments/exp043/codes/synthetic_freq_blend.py
group: experiments/exp043/codes
experiment: exp043
loc: 59
tags: [code, exp043]
---

# synthetic_freq_blend

> Synthetic between-anchor heatmap blending for multifreq training.

**Source:** `experiments/exp043/codes/synthetic_freq_blend.py` · 59 lines
**Experiment:** [[exp043]]

## Purpose

```text
Synthetic between-anchor heatmap blending for multifreq training.

Purpose:
    Training augmentation: interpolate heatmap + PI_freq between anchor pairs for off-anchor supervision.

Run:
    Import only — called from ``train_vae_simple._prepare_batch`` when ``synthetic_blend_prob > 0``.

Agent notes:
    - What: Stochastic freq/heatmap blend augmentation during VAE training.
    - Usage: Set ``synthetic_blend_prob`` in experiment ``config.yaml``; batch must include ``heatmap_norm_alt``, ``PI_freq_alt``.
    - Key symbol: ``maybe_apply_synthetic_blend(batch, cfg)``
```

## Functions

- **`maybe_apply_synthetic_blend(batch: dict, cfg)`**

## Imported by

- [[experiments.exp043.codes.train_vae_simple]]
- [[gmax_training_patch]]

## External dependencies

`torch`
