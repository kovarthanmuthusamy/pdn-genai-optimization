---
title: vae_multi_input_simple
type: code
path: experiments/exp037_lat_change/codes/vae_multi_input_simple.py
group: experiments/exp037_lat_change/codes
experiment: exp037_lat_change
loc: 491
tags: [code, exp037_lat_change]
---

# vae_multi_input_simple

**Source:** `experiments/exp037_lat_change/codes/vae_multi_input_simple.py` · 491 lines
**Experiment:** [[exp037_lat_change]]

## Constants

| Name | Value |
|------|-------|
| `IMP_DROPOUT` | `0.1` |

## Classes

- **`DeepMLP(nn.Module)`** — Deep MLP with LayerNorm, SiLU, dropout, and residual skip connections.
- **`SelfAttn2d(nn.Module)`** — Lightweight spatial self-attention for 2D feature maps (B, C, H, W).
- **`MultiInputVAE(nn.Module)`** — Multi-modal VAE with a shared latent space + heatmap-private tail dims.

## Imported by

- [[experiments.exp037_lat_change.codes.inference_vae]]
- [[experiments.exp037_lat_change.codes.train_vae_simple]]

## External dependencies

`torch`
