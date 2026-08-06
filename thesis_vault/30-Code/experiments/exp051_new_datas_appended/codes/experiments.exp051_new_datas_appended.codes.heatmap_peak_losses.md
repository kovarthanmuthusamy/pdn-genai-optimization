---
title: heatmap_peak_losses
type: code
path: experiments/exp051_new_datas_appended/codes/heatmap_peak_losses.py
group: experiments/exp051_new_datas_appended/codes
experiment: exp051_new_datas_appended
loc: 155
tags: [code, exp051_new_datas_appended]
---

# heatmap_peak_losses

> Tier A heatmap losses for exp051.

**Source:** `experiments/exp051_new_datas_appended/codes/heatmap_peak_losses.py` · 155 lines
**Experiment:** [[exp051_new_datas_appended]]

## Purpose

```text
Tier A heatmap losses for exp051.

FG huber + FG-masked gradient vector/direction + physical top-k blob (Ω).
Background contributes zero to gradient terms. No percentile training.
```

## Functions

- **`_sobel_kernels(device: torch.device, dtype: torch.dtype)`**
- **`_sobel_grad(hm: torch.Tensor)`**
- **`_masked_mean(per_px: torch.Tensor, mask: torch.Tensor)`**
- **`grad_vector_field_loss(recon: torch.Tensor, target: torch.Tensor, fg: torch.Tensor, c: _tr.Config)`** — FG-masked ∇ loss: vector huber + flow direction (background = 0).
- **`_topk_target_mask(target: torch.Tensor, fg: torch.Tensor, bg: float, k: int)`**
- **`_norm_to_physical_maps(recon: torch.Tensor, target: torch.Tensor, c: _tr.Config, *, pi_freq: torch.Tensor | None)`**
- **`_blob_under_over(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, *, overshoot_weight: float)`**
- **`heatmap_loss_tier_a(recon: torch.Tensor, target: torch.Tensor, c: _tr.Config)`** — FG huber + grad vector/direction — per-sample (B,).
- **`heatmap_peak_blob_phys_loss(recon: torch.Tensor, target: torch.Tensor, c: _tr.Config, *, pi_freq: torch.Tensor | None=None)`** — Physical Ω top-k blob: under-predict + overshoot (scalar mean).

## Imports

- [[experiments.exp038_true_multi.codes.train_vae_simple]]
- [[norm_stats]]

## Imported by

- [[experiments.exp051_new_datas_appended.codes.run_epoch_encode]]
- [[experiments.exp051_new_datas_appended.codes.train_vae_simple]]

## External dependencies

`src_vae`, `torch`
