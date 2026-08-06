---
title: vae_poe_freq
type: code
path: experiments/exp054_K_30/codes/vae_poe_freq.py
group: experiments/exp054_K_30/codes
experiment: exp054_K_30
loc: 295
tags: [code, exp054_K_30]
---

# vae_poe_freq

> Multi-input VAE — exp045: U-Net skips + freq PoE private dims + encode-first.

**Source:** `experiments/exp054_K_30/codes/vae_poe_freq.py` · 295 lines
**Experiment:** [[exp054_K_30]]

## Classes

- **`MultiInputVAEPoeFreq(MultiInputVAE)`** — PoE fusion with heatmap U-Net skips and PI_freq expert on private dims.

## Imports

- [[experiments.exp054_K_30.codes.vae_multi_input_simple]]

## Imported by

- [[experiments.exp054_K_30.codes.inference_vae]]
- [[experiments.exp054_K_30.codes.train_core]]
- [[experiments.exp054_K_30.codes.train_vae_simple]]

## External dependencies

`torch`
