---
title: vae_loss_backup
type: code
path: src_vae/loss/vae_loss_backup.py
group: src_vae/loss
loc: 349
tags: [code, src_vae]
---

# vae_loss_backup

> Extended multi-input VAE loss (backup / reference implementation).

**Source:** `src_vae/loss/vae_loss_backup.py` · 349 lines

## Purpose

```text
Extended multi-input VAE loss (backup / reference implementation).

Run: ``from src_vae.loss.vae_loss_backup import VAELoss`` when experimenting with extra losses.
```

## Classes

- **`VAELoss(nn.Module)`** — Combined loss function for multi-output VAE with Uncertainty-Based Weighting (Kendall et al.)

## Functions

- **`dice_loss(pred_logits: torch.Tensor, target: torch.Tensor, smooth: float=1e-06)`** — Dice Loss for binary segmentation (occupancy maps)
- **`focal_loss(pred_logits: torch.Tensor, target: torch.Tensor, alpha: float=0.25, gamma: float=2.0, reduction: str='mean')`** — Focal Loss for handling class imbalance in sparse occupancy grids.
- **`cosine_similarity_loss(pred: torch.Tensor, target: torch.Tensor, eps: float=1e-08)`** — Cosine Similarity Loss for impedance vectors (shape & direction)
- **`ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int=11, size_average: bool=True)`** — Structural Similarity Index (SSIM) Loss for heatmaps

## External dependencies

`torch`
