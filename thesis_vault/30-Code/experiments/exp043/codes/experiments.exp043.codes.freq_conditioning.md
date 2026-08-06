---
title: freq_conditioning
type: code
path: experiments/exp043/codes/freq_conditioning.py
group: experiments/exp043/codes
experiment: exp043
loc: 60
tags: [code, exp043]
---

# freq_conditioning

> PI frequency conditioning: Fourier features + MLP; FiLM for heatmap conv blocks.

**Source:** `experiments/exp043/codes/freq_conditioning.py` · 60 lines
**Experiment:** [[exp043]]

## Classes

- **`FreqConditioner(nn.Module)`** — Map normalised PI_freq in [0, 1] to cond_dim (replaces Linear(1, cond_dim)).
- **`FiLM2d(nn.Module)`** — Feature-wise scale/shift from a conditioning vector.

## Imported by

- [[experiments.exp043.codes.vae_multi_input_simple]]
- [[experiments.exp045.codes.vae_multi_input_simple]]
- [[experiments.exp046.codes.vae_multi_input_simple]]
- [[experiments.exp047.codes.vae_multi_input_simple]]
- [[experiments.exp048.codes.vae_multi_input_simple]]
- [[experiments.exp049.codes.vae_multi_input_simple]]
- [[experiments.exp050.codes.vae_multi_input_simple]]
- [[experiments.exp051_new_datas_appended.codes.vae_multi_input_simple]]
- [[experiments.exp052_unbounded_pearson.codes.vae_multi_input_simple]]
- [[experiments.exp053_peak_log1p_losses.codes.vae_multi_input_simple]]
- [[experiments.exp054_K_30.codes.vae_multi_input_simple]]
- [[experiments.exp055_hard_occ.codes.vae_multi_input_simple]]
- [[experiments.exp056_graph_vae.codes.vae_multi_input_simple]]
- [[experiments.exp057_structured_graph.codes.vae_multi_input_simple]]
- [[experiments.exp058_asymmetric_kl.codes.vae_multi_input_simple]]
- [[experiments.exp059_capacity_freq.codes.vae_multi_input_simple]]
- [[experiments.exp060_multitype_occ.codes.vae_multi_input_simple]]

## External dependencies

`torch`
