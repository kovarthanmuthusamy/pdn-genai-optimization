---
title: vae_loss
type: code
path: src_vae/loss/vae_loss.py
group: src_vae/loss
loc: 211
tags: [code, src_vae]
---

# vae_loss

> Active multi-input VAE loss with uncertainty weighting.

**Source:** `src_vae/loss/vae_loss.py` · 211 lines

## Purpose

```text
Active multi-input VAE loss with uncertainty weighting.

Run: ``from src_vae.loss.vae_loss import VAELoss`` in experiment training loops.
```

## Classes

- **`VAELoss(nn.Module)`** — VAE loss with uncertainty-based weighting and KL divergence

## Functions

- **`dice_loss(pred_logits: torch.Tensor, target: torch.Tensor, smooth: float=1e-06)`** — Dice Loss for binary segmentation

## External dependencies

`torch`
