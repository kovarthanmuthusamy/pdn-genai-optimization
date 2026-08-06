---
title: heatmap_peak_losses
type: code
path: experiments/exp052_unbounded_pearson/codes/heatmap_peak_losses.py
group: experiments/exp052_unbounded_pearson/codes
experiment: exp052_unbounded_pearson
loc: 168
tags: [code, exp052_unbounded_pearson]
---

# heatmap_peak_losses

> exp052 heatmap losses: FG Pearson (global) + gradient field (local) + log1p peak blob.

**Source:** `experiments/exp052_unbounded_pearson/codes/heatmap_peak_losses.py` · 168 lines
**Experiment:** [[exp052_unbounded_pearson]]

## Purpose

```text
exp052 heatmap losses: FG Pearson (global) + gradient field (local) + log1p peak blob.

Phys blob runs in robust log1p space (not denormed Ω) so unbounded z cannot explode via exp().
No percentile / dynrange / Huber FG MSE terms.
```

## Functions

- **`_sobel_kernels(device: torch.device, dtype: torch.dtype)`**
- **`_sobel_grad(hm: torch.Tensor)`**
- **`_masked_mean(per_px: torch.Tensor, mask: torch.Tensor)`**
- **`grad_vector_field_loss(recon: torch.Tensor, target: torch.Tensor, fg: torch.Tensor, c: _tr.Config)`** — FG-masked grad loss: vector huber + flow direction.
- **`_topk_target_mask(target: torch.Tensor, fg: torch.Tensor, bg: float, k: int)`**
- **`_robust_log1p_maps(recon: torch.Tensor, target: torch.Tensor, c: _tr.Config, *, pi_freq: torch.Tensor | None)`** — Map z-score heatmaps to log1p(Ω) per MHz bin (linear; no exp blow-up).
- **`_blob_under_over(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, *, overshoot_weight: float)`**
- **`heatmap_loss_pearson_grad(recon: torch.Tensor, target: torch.Tensor, c: _tr.Config, *, lite: bool=False)`** — FG Pearson (global) + grad field (local) — per-sample (B,).
- **`heatmap_peak_blob_phys_loss(recon: torch.Tensor, target: torch.Tensor, c: _tr.Config, *, pi_freq: torch.Tensor | None=None, reduction: str='mean')`** — Top-k peak blob in robust log1p space: under-predict + overshoot.

## Imports

- [[experiments.exp038_true_multi.codes.train_vae_simple]]
- [[experiments.exp052_unbounded_pearson.codes.spatial_metrics]]
- [[norm_stats]]

## Imported by

- [[experiments.exp052_unbounded_pearson.codes.diagnose_nan_grad]]
- [[experiments.exp052_unbounded_pearson.codes.run_epoch_encode]]
- [[experiments.exp052_unbounded_pearson.codes.train_vae_simple]]

## External dependencies

`src_vae`, `torch`
