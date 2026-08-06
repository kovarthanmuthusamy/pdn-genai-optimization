---
title: vae_multi_input_simple
type: code
path: experiments/exp045/codes/vae_multi_input_simple.py
group: experiments/exp045/codes
experiment: exp045
loc: 926
tags: [code, exp045]
---

# vae_multi_input_simple

**Source:** `experiments/exp045/codes/vae_multi_input_simple.py` · 926 lines
**Experiment:** [[exp045]]

## Constants

| Name | Value |
|------|-------|
| `HM_BOTTLENECK` | `8` |
| `HM_DROPOUT` | `0.05` |
| `HM_OUT_SCALE` | `1.02` |
| `IMP_DROPOUT` | `0.15` |

## Classes

- **`DeepMLP(nn.Module)`** — Deep MLP with LayerNorm, SiLU activations, and optional dropout.
- **`SelfAttn2d(nn.Module)`** — Lightweight spatial self-attention for 2D feature maps (B, C, H, W).
- **`ResidualConv2d(nn.Module)`** — Residual conv block at fixed spatial resolution.
- **`UpsampleConv2d(nn.Module)`** — Bilinear upsample + conv — avoids checkerboard artifacts from ConvTranspose2d.
- **`OptionalSkipFuse(nn.Module)`** — Fuse decoder feature map with an optional encoder skip (U-Net).
- **`OccSpatialTower(nn.Module)`** — 7×8 pad occupancy grid → multi-scale layout features for decoder fusion.
- **`MultiInputVAE(nn.Module)`** — Multi-modal VAE with a shared latent space + heatmap-private tail dims.

## Imports

- [[experiments.exp043.codes.freq_conditioning]]
- [[experiments.exp043.codes.freq_inference_utils]]
- [[occupancy]]
- [[pi_freq_utils]]

## Imported by

- [[experiments.exp045.codes.vae_poe_freq]]

## External dependencies

`libs`, `src_vae`, `torch`
