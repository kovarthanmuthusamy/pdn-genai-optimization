---
title: vae_multi_input_simple
type: code
path: experiments/exp055_hard_occ/codes/vae_multi_input_simple.py
group: experiments/exp055_hard_occ/codes
experiment: exp055_hard_occ
loc: 978
tags: [code, exp055_hard_occ]
---

# vae_multi_input_simple

**Source:** `experiments/exp055_hard_occ/codes/vae_multi_input_simple.py` · 978 lines
**Experiment:** [[exp055_hard_occ]]

## Constants

| Name | Value |
|------|-------|
| `HM_BOTTLENECK` | `8` |
| `HM_DROPOUT` | `0.05` |
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
- [[experiments.exp055_hard_occ.codes.occupancy_binary]]
- [[occupancy]]
- [[pi_freq_utils]]

## Imported by

- [[experiments.exp055_hard_occ.codes.vae_poe_freq]]

## External dependencies

`libs`, `src_vae`, `torch`
