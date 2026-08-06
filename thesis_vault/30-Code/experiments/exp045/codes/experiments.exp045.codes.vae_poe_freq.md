---
title: vae_poe_freq
type: code
path: experiments/exp045/codes/vae_poe_freq.py
group: experiments/exp045/codes
experiment: exp045
loc: 235
tags: [code, exp045]
---

# vae_poe_freq

> Multi-input VAE — exp045: U-Net skips + freq PoE private dims + encode-first.

**Source:** `experiments/exp045/codes/vae_poe_freq.py` · 235 lines
**Experiment:** [[exp045]]

## Classes

- **`MultiInputVAEPoeFreq(MultiInputVAE)`** — PoE fusion with heatmap U-Net skips and PI_freq expert on private dims.

## Imports

- [[experiments.exp045.codes.vae_multi_input_simple]]

## Imported by

- [[exp045_eval_common]]
- [[experiments.exp045.codes.inference_vae]]
- [[experiments.exp045.codes.train_vae_simple]]

## External dependencies

`torch`
