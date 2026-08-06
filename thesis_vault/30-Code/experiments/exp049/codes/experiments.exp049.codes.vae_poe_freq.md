---
title: vae_poe_freq
type: code
path: experiments/exp049/codes/vae_poe_freq.py
group: experiments/exp049/codes
experiment: exp049
loc: 295
tags: [code, exp049]
---

# vae_poe_freq

> Multi-input VAE — exp045: U-Net skips + freq PoE private dims + encode-first.

**Source:** `experiments/exp049/codes/vae_poe_freq.py` · 295 lines
**Experiment:** [[exp049]]

## Classes

- **`MultiInputVAEPoeFreq(MultiInputVAE)`** — PoE fusion with heatmap U-Net skips and PI_freq expert on private dims.

## Imports

- [[experiments.exp049.codes.vae_multi_input_simple]]

## Imported by

- [[exp049_eval_common]]
- [[experiments.exp049.codes.inference_vae]]
- [[experiments.exp049.codes.train_vae_simple]]

## External dependencies

`torch`
