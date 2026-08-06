---
title: vae_poe_freq
type: code
path: experiments/exp058_asymmetric_kl/codes/vae_poe_freq.py
group: experiments/exp058_asymmetric_kl/codes
experiment: exp058_asymmetric_kl
loc: 305
tags: [code, exp058_asymmetric_kl, uncommitted]
---

# vae_poe_freq

> Multi-input VAE — exp045: U-Net skips + freq PoE private dims + encode-first.

**Source:** `experiments/exp058_asymmetric_kl/codes/vae_poe_freq.py` · 305 lines
**Git:** uncommitted — not yet tracked
**Experiment:** [[exp058_asymmetric_kl]]

## Classes

- **`MultiInputVAEPoeFreq(MultiInputVAE)`** — PoE fusion with heatmap U-Net skips and PI_freq expert on private dims.

## Imports

- [[experiments.exp058_asymmetric_kl.codes.vae_multi_input_simple]]

## Imported by

- [[experiments.exp058_asymmetric_kl.codes.inference_vae]]
- [[experiments.exp058_asymmetric_kl.codes.train_core]]
- [[experiments.exp058_asymmetric_kl.codes.train_vae_simple]]

## External dependencies

`torch`
