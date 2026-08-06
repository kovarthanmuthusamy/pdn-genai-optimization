---
title: vae_poe_freq
type: code
path: experiments/exp050/codes/vae_poe_freq.py
group: experiments/exp050/codes
experiment: exp050
loc: 295
tags: [code, exp050]
---

# vae_poe_freq

> Multi-input VAE — exp045: U-Net skips + freq PoE private dims + encode-first.

**Source:** `experiments/exp050/codes/vae_poe_freq.py` · 295 lines
**Experiment:** [[exp050]]

## Classes

- **`MultiInputVAEPoeFreq(MultiInputVAE)`** — PoE fusion with heatmap U-Net skips and PI_freq expert on private dims.

## Imports

- [[experiments.exp050.codes.vae_multi_input_simple]]

## Imported by

- [[exp050_eval_common]]
- [[experiments.exp050.codes.inference_vae]]
- [[experiments.exp050.codes.train_vae_simple]]

## External dependencies

`torch`
