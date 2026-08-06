---
title: freq_conditioning
type: code
path: experiments/exp040/codes/codes/freq_conditioning.py
group: experiments/exp040/codes/codes
experiment: exp040
loc: 60
tags: [code, exp040]
---

# freq_conditioning

> PI frequency conditioning: Fourier features + MLP; FiLM for heatmap conv blocks.

**Source:** `experiments/exp040/codes/codes/freq_conditioning.py` · 60 lines
**Experiment:** [[exp040]]

## Classes

- **`FreqConditioner(nn.Module)`** — Map normalised PI_freq in [0, 1] to cond_dim (replaces Linear(1, cond_dim)).
- **`FiLM2d(nn.Module)`** — Feature-wise scale/shift from a conditioning vector.

## External dependencies

`torch`
