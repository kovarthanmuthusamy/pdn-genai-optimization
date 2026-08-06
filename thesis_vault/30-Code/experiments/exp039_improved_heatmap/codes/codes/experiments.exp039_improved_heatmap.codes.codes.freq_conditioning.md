---
title: freq_conditioning
type: code
path: experiments/exp039_improved_heatmap/codes/codes/freq_conditioning.py
group: experiments/exp039_improved_heatmap/codes/codes
experiment: exp039_improved_heatmap
loc: 60
tags: [code, exp039_improved_heatmap]
---

# freq_conditioning

> PI frequency conditioning: Fourier features + MLP; FiLM for heatmap conv blocks.

**Source:** `experiments/exp039_improved_heatmap/codes/codes/freq_conditioning.py` · 60 lines
**Experiment:** [[exp039_improved_heatmap]]

## Classes

- **`FreqConditioner(nn.Module)`** — Map normalised PI_freq in [0, 1] to cond_dim (replaces Linear(1, cond_dim)).
- **`FiLM2d(nn.Module)`** — Feature-wise scale/shift from a conditioning vector.

## External dependencies

`torch`
