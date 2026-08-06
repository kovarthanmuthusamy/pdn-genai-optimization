---
title: synthetic_freq_blend
type: code
path: experiments/exp040/codes/synthetic_freq_blend.py
group: experiments/exp040/codes
experiment: exp040
loc: 54
tags: [code, exp040]
---

# synthetic_freq_blend

> Synthetic between-anchor heatmap targets for multifreq training (exp040).

**Source:** `experiments/exp040/codes/synthetic_freq_blend.py` · 54 lines
**Experiment:** [[exp040]]

## Functions

- **`maybe_apply_synthetic_blend(batch: dict, cfg)`** — With probability ``synthetic_blend_prob``, replace (heatmap, PI_freq) by a

## Imported by

- [[experiments.exp040.codes.train_vae_simple]]

## External dependencies

`torch`
