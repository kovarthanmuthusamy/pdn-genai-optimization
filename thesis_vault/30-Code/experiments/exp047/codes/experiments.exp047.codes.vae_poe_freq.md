---
title: vae_poe_freq
type: code
path: experiments/exp047/codes/vae_poe_freq.py
group: experiments/exp047/codes
experiment: exp047
loc: 295
tags: [code, exp047]
---

# vae_poe_freq

> Multi-input VAE — exp045: U-Net skips + freq PoE private dims + encode-first.

**Source:** `experiments/exp047/codes/vae_poe_freq.py` · 295 lines
**Experiment:** [[exp047]]

## Classes

- **`MultiInputVAEPoeFreq(MultiInputVAE)`** — PoE fusion with heatmap U-Net skips and PI_freq expert on private dims.

## Imports

- [[experiments.exp047.codes.vae_multi_input_simple]]

## Imported by

- [[exp047_eval_common]]
- [[experiments.exp047.codes.inference_vae]]
- [[experiments.exp047.codes.train_vae_simple]]

## External dependencies

`torch`
