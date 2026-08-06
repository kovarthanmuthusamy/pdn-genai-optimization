---
title: vae_multi_input_simple
type: code
path: experiments/exp039_improved_heatmap/codes/codes/vae_multi_input_simple.py
group: experiments/exp039_improved_heatmap/codes/codes
experiment: exp039_improved_heatmap
loc: 693
tags: [code, exp039_improved_heatmap]
---

# vae_multi_input_simple

**Source:** `experiments/exp039_improved_heatmap/codes/codes/vae_multi_input_simple.py` · 693 lines
**Experiment:** [[exp039_improved_heatmap]]

## Constants

| Name | Value |
|------|-------|
| `IMP_DROPOUT` | `0.15` |

## Classes

- **`DeepMLP(nn.Module)`** — Deep MLP with LayerNorm, SiLU activations, and optional dropout.
- **`SelfAttn2d(nn.Module)`** — Lightweight spatial self-attention for 2D feature maps (B, C, H, W).
- **`MultiInputVAE(nn.Module)`** — Multi-modal VAE with a shared latent space + heatmap-private tail dims.

## Imports

- [[experiments.exp038_true_multi.codes.freq_conditioning]]
- [[experiments.exp038_true_multi.codes.freq_inference_utils]]
- [[pi_freq_utils]]

## External dependencies

`src_vae`, `torch`
