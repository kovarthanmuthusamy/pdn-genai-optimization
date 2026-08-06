---
title: vae_multi_input
type: code
path: src_vae/model/vae_multi_input.py
group: src_vae/model
loc: 1229
tags: [code, src_vae]
---

# vae_multi_input

> Multi-input VAE with mid-layer fusion and hierarchical decoder.

**Source:** `src_vae/model/vae_multi_input.py` · 1229 lines

## Purpose

```text
Multi-input VAE with mid-layer fusion and hierarchical decoder.

Run: ``from src_vae.model.vae_multi_input import MultiInputVAE`` in experiment training scripts.
```

## Constants

| Name | Value |
|------|-------|
| `ATTENTION_GAMMA_INIT` | `0.05` |
| `OCCUPANCY_ATTENTION_GAMMA_INIT` | `0.15` |

## Classes

- **`SelfAttention2D(nn.Module)`** — Self-Attention for 2D feature maps - captures global spatial relationships
- **`SelfAttention1D(nn.Module)`** — Self-Attention for 1D sequences - captures relationships in vector features
- **`HeatmapBranch(nn.Module)`** — Heatmap encoder branch: 64x64x2 → 8x8x128
- **`OccupancyBranch(nn.Module)`** — Occupancy encoder branch: 7x8x1 → 8x8x128
- **`ImpedanceBranch(nn.Module)`** — Impedance encoder branch: 231x1 → 8x8x128 (via MLP then reshape)
- **`MaxImpedanceBranch(nn.Module)`** — Max Impedance encoder branch: 1 (scalar) → 8x8x128 (via MLP then reshape)
- **`LateFusionEncoder(nn.Module)`** — 🔥 LATE FUSION ENCODER: Preserves modality signal strength via late fusion
- **`MidLayerFusionEncoder(nn.Module)`** — Mid-layer fusion encoder: fuses modalities at 8x8 resolution before self-attention
- **`Encoder(nn.Module)`** — Multi-Input VAE Encoder with selectable fusion strategies
- **`MasterFeatureGridDecoder(nn.Module)`** — Shared master feature grid decoder: latent vector → 16x16x128 spatial features
- **`HeatmapDecoder(nn.Module)`** — Decoder for 64x64x2 heatmap - independent decoder head
- **`OccupancyDecoder(nn.Module)`** — Decoder for 7x8x1 binary occupancy map from shared master grid
- **`ImpedanceDecoder(nn.Module)`** — Decoder for 231x1 impedance vector
- **`MaxImpedanceDecoder(nn.Module)`** — Decoder for max impedance scalar with STABILITY-FIRST design
- **`Decoder(nn.Module)`** — Hierarchical decoder with STABILITY-FIRST design
- **`MultiInputVAE(nn.Module)`** — Variational Autoencoder with STABILITY-FIRST physically-aware design.

## Functions

- **`_init_weights(module)`** — Shared weight initialization to prevent NaN
- **`_check_nan_inf(tensor: torch.Tensor, name: str, input_tensor: torch.Tensor)`** — Check for NaN/Inf in tensors and raise descriptive error

## External dependencies

`torch`
