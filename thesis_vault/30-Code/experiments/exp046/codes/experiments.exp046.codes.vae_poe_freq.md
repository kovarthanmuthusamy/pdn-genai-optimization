---
title: vae_poe_freq
type: code
path: experiments/exp046/codes/vae_poe_freq.py
group: experiments/exp046/codes
experiment: exp046
loc: 287
tags: [code, exp046]
---

# vae_poe_freq

> Multi-input VAE — exp045: U-Net skips + freq PoE private dims + encode-first.

**Source:** `experiments/exp046/codes/vae_poe_freq.py` · 287 lines
**Experiment:** [[exp046]]

## Classes

- **`MultiInputVAEPoeFreq(MultiInputVAE)`** — PoE fusion with heatmap U-Net skips and PI_freq expert on private dims.

## Imports

- [[experiments.exp046.codes.vae_multi_input_simple]]

## Imported by

- [[exp046_eval_common]]
- [[experiments.exp046.codes.inference_vae]]
- [[experiments.exp046.codes.train_vae_simple]]

## External dependencies

`torch`
