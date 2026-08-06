---
title: vae_multi_input_simple
type: code
path: experiments/exp060_multitype_occ/codes/vae_multi_input_simple.py
group: experiments/exp060_multitype_occ/codes
experiment: exp060_multitype_occ
loc: 1131
tags: [code, exp060_multitype_occ, uncommitted]
---

# vae_multi_input_simple

**Source:** `experiments/exp060_multitype_occ/codes/vae_multi_input_simple.py` · 1131 lines
**Git:** uncommitted — not yet tracked
**Experiment:** [[exp060_multitype_occ]]

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
- **`OccSpatialTower(nn.Module)`** — 7×8 pad occupancy grid → multi-scale layout features for decoder fusion.
- **`MultiInputVAE(nn.Module)`** — Multi-modal VAE with a shared latent space + heatmap-private tail dims.

## Imports

- [[experiments.exp043.codes.freq_conditioning]]
- [[experiments.exp043.codes.freq_inference_utils]]
- [[experiments.exp060_multitype_occ.codes.graph_imp]]
- [[experiments.exp060_multitype_occ.codes.graph_occ]]
- [[occupancy]]
- [[occupancy_types]]
- [[pi_freq_utils]]

## Imported by

- [[experiments.exp060_multitype_occ.codes.vae_poe_freq]]

## External dependencies

`libs`, `src_vae`, `torch`
