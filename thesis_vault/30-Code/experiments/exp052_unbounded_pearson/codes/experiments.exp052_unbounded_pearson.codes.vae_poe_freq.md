---
title: vae_poe_freq
type: code
path: experiments/exp052_unbounded_pearson/codes/vae_poe_freq.py
group: experiments/exp052_unbounded_pearson/codes
experiment: exp052_unbounded_pearson
loc: 295
tags: [code, exp052_unbounded_pearson]
---

# vae_poe_freq

> Multi-input VAE — exp045: U-Net skips + freq PoE private dims + encode-first.

**Source:** `experiments/exp052_unbounded_pearson/codes/vae_poe_freq.py` · 295 lines
**Experiment:** [[exp052_unbounded_pearson]]

## Classes

- **`MultiInputVAEPoeFreq(MultiInputVAE)`** — PoE fusion with heatmap U-Net skips and PI_freq expert on private dims.

## Imports

- [[experiments.exp052_unbounded_pearson.codes.vae_multi_input_simple]]

## Imported by

- [[exp052_eval_common]]
- [[experiments.exp052_unbounded_pearson.codes.inference_vae]]
- [[experiments.exp052_unbounded_pearson.codes.train_vae_simple]]

## External dependencies

`torch`
