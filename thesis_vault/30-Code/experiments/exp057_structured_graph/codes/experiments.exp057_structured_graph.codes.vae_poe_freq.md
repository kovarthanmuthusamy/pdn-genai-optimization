---
title: vae_poe_freq
type: code
path: experiments/exp057_structured_graph/codes/vae_poe_freq.py
group: experiments/exp057_structured_graph/codes
experiment: exp057_structured_graph
loc: 305
tags: [code, exp057_structured_graph]
---

# vae_poe_freq

> Multi-input VAE — exp045: U-Net skips + freq PoE private dims + encode-first.

**Source:** `experiments/exp057_structured_graph/codes/vae_poe_freq.py` · 305 lines
**Experiment:** [[exp057_structured_graph]]

## Classes

- **`MultiInputVAEPoeFreq(MultiInputVAE)`** — PoE fusion with heatmap U-Net skips and PI_freq expert on private dims.

## Imports

- [[experiments.exp057_structured_graph.codes.vae_multi_input_simple]]

## Imported by

- [[experiments.exp057_structured_graph.codes.inference_vae]]
- [[experiments.exp057_structured_graph.codes.train_core]]
- [[experiments.exp057_structured_graph.codes.train_vae_simple]]

## External dependencies

`torch`
