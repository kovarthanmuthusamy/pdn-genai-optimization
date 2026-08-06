---
title: vae_poe_freq
type: code
path: experiments/exp053_peak_log1p_losses/codes/vae_poe_freq.py
group: experiments/exp053_peak_log1p_losses/codes
experiment: exp053_peak_log1p_losses
loc: 295
tags: [code, exp053_peak_log1p_losses]
---

# vae_poe_freq

> Multi-input VAE — exp045: U-Net skips + freq PoE private dims + encode-first.

**Source:** `experiments/exp053_peak_log1p_losses/codes/vae_poe_freq.py` · 295 lines
**Experiment:** [[exp053_peak_log1p_losses]]

## Classes

- **`MultiInputVAEPoeFreq(MultiInputVAE)`** — PoE fusion with heatmap U-Net skips and PI_freq expert on private dims.

## Imports

- [[experiments.exp053_peak_log1p_losses.codes.vae_multi_input_simple]]

## Imported by

- [[exp053_eval_common]]
- [[experiments.exp053_peak_log1p_losses.codes.inference_vae]]
- [[experiments.exp053_peak_log1p_losses.codes.train_vae_simple]]

## External dependencies

`torch`
