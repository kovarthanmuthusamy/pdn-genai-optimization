---
title: vae_poe_freq
type: code
path: experiments/exp048/codes/vae_poe_freq.py
group: experiments/exp048/codes
experiment: exp048
loc: 295
tags: [code, exp048]
---

# vae_poe_freq

> Multi-input VAE — exp045: U-Net skips + freq PoE private dims + encode-first.

**Source:** `experiments/exp048/codes/vae_poe_freq.py` · 295 lines
**Experiment:** [[exp048]]

## Classes

- **`MultiInputVAEPoeFreq(MultiInputVAE)`** — PoE fusion with heatmap U-Net skips and PI_freq expert on private dims.

## Imports

- [[experiments.exp048.codes.vae_multi_input_simple]]

## Imported by

- [[exp048_eval_common]]
- [[experiments.exp048.codes.inference_vae]]
- [[experiments.exp048.codes.train_vae_simple]]

## External dependencies

`torch`
