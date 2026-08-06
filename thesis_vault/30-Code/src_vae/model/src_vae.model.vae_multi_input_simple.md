---
title: vae_multi_input_simple
type: code
path: src_vae/model/vae_multi_input_simple.py
group: src_vae/model
loc: 572
tags: [code, src_vae]
---

# vae_multi_input_simple

> Simplified multi-input VAE (heatmap + occupancy + impedance).

**Source:** `src_vae/model/vae_multi_input_simple.py` · 572 lines

## Purpose

```text
Simplified multi-input VAE (heatmap + occupancy + impedance).

Run: ``from src_vae.model.vae_multi_input_simple import MultiInputVAE`` in training scripts.
```

## Classes

- **`SE1d(nn.Module)`** — Squeeze-and-Excitation block for 1D feature maps (B, C, L).
- **`MultiInputVAE(nn.Module)`**

## External dependencies

`torch`
