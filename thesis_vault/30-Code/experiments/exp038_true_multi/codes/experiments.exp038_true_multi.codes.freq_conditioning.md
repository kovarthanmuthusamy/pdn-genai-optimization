---
title: freq_conditioning
type: code
path: experiments/exp038_true_multi/codes/freq_conditioning.py
group: experiments/exp038_true_multi/codes
experiment: exp038_true_multi
loc: 60
tags: [code, exp038_true_multi]
---

# freq_conditioning

> PI frequency conditioning: Fourier features + MLP; FiLM for heatmap conv blocks.

**Source:** `experiments/exp038_true_multi/codes/freq_conditioning.py` · 60 lines
**Experiment:** [[exp038_true_multi]]

## Classes

- **`FreqConditioner(nn.Module)`** — Map normalised PI_freq in [0, 1] to cond_dim (replaces Linear(1, cond_dim)).
- **`FiLM2d(nn.Module)`** — Feature-wise scale/shift from a conditioning vector.

## Imported by

- [[experiments.exp038_true_multi.codes.vae_multi_input_simple]]
- [[experiments.exp039_improved_heatmap.codes.codes.vae_multi_input_simple]]
- [[experiments.exp040.codes.codes.vae_multi_input_simple]]
- [[experiments.exp041.codes.codes.vae_multi_input_simple]]
- [[vae_factorized_freq]]

## External dependencies

`torch`
