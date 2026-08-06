---
title: synthetic_freq_blend
type: code
path: experiments/exp039_improved_heatmap/codes/synthetic_freq_blend.py
group: experiments/exp039_improved_heatmap/codes
experiment: exp039_improved_heatmap
loc: 52
tags: [code, exp039_improved_heatmap]
---

# synthetic_freq_blend

> Synthetic between-anchor heatmap targets for multifreq training (exp039).

**Source:** `experiments/exp039_improved_heatmap/codes/synthetic_freq_blend.py` · 52 lines
**Experiment:** [[exp039_improved_heatmap]]

## Functions

- **`maybe_apply_synthetic_blend(batch: dict, cfg)`** — With probability ``synthetic_blend_prob``, replace (heatmap, PI_freq) by a

## Imported by

- [[experiments.exp039_improved_heatmap.codes.train_vae_simple]]

## External dependencies

`torch`
