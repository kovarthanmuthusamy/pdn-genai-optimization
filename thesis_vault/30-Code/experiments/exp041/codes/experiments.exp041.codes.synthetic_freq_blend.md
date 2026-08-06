---
title: synthetic_freq_blend
type: code
path: experiments/exp041/codes/synthetic_freq_blend.py
group: experiments/exp041/codes
experiment: exp041
loc: 52
tags: [code, exp041]
---

# synthetic_freq_blend

> Synthetic between-anchor heatmap targets for multifreq training (exp041).

**Source:** `experiments/exp041/codes/synthetic_freq_blend.py` · 52 lines
**Experiment:** [[exp041]]

## Functions

- **`maybe_apply_synthetic_blend(batch: dict, cfg)`** — With probability ``synthetic_blend_prob``, replace (heatmap, PI_freq) by a

## Imported by

- [[experiments.exp041.codes.train_vae_simple]]

## External dependencies

`torch`
