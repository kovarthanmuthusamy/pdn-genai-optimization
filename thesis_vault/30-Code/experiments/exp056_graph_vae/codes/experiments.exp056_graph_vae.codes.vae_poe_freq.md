---
title: vae_poe_freq
type: code
path: experiments/exp056_graph_vae/codes/vae_poe_freq.py
group: experiments/exp056_graph_vae/codes
experiment: exp056_graph_vae
loc: 301
tags: [code, exp056_graph_vae]
---

# vae_poe_freq

> Multi-input VAE — exp045: U-Net skips + freq PoE private dims + encode-first.

**Source:** `experiments/exp056_graph_vae/codes/vae_poe_freq.py` · 301 lines
**Experiment:** [[exp056_graph_vae]]

## Classes

- **`MultiInputVAEPoeFreq(MultiInputVAE)`** — PoE fusion with heatmap U-Net skips and PI_freq expert on private dims.

## Imports

- [[experiments.exp056_graph_vae.codes.vae_multi_input_simple]]

## Imported by

- [[experiments.exp056_graph_vae.codes.inference_vae]]
- [[experiments.exp056_graph_vae.codes.train_core]]
- [[experiments.exp056_graph_vae.codes.train_vae_simple]]

## External dependencies

`torch`
