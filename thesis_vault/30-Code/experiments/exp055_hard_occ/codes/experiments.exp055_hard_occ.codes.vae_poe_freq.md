---
title: vae_poe_freq
type: code
path: experiments/exp055_hard_occ/codes/vae_poe_freq.py
group: experiments/exp055_hard_occ/codes
experiment: exp055_hard_occ
loc: 295
tags: [code, exp055_hard_occ]
---

# vae_poe_freq

> Multi-input VAE — exp045: U-Net skips + freq PoE private dims + encode-first.

**Source:** `experiments/exp055_hard_occ/codes/vae_poe_freq.py` · 295 lines
**Experiment:** [[exp055_hard_occ]]

## Classes

- **`MultiInputVAEPoeFreq(MultiInputVAE)`** — PoE fusion with heatmap U-Net skips and PI_freq expert on private dims.

## Imports

- [[experiments.exp055_hard_occ.codes.vae_multi_input_simple]]

## Imported by

- [[experiments.exp055_hard_occ.codes.inference_vae]]
- [[experiments.exp055_hard_occ.codes.train_core]]
- [[experiments.exp055_hard_occ.codes.train_vae_simple]]

## External dependencies

`torch`
