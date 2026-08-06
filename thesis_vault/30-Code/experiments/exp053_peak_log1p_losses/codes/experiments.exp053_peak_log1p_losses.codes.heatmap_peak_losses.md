---
title: heatmap_peak_losses
type: code
path: experiments/exp053_peak_log1p_losses/codes/heatmap_peak_losses.py
group: experiments/exp053_peak_log1p_losses/codes
experiment: exp053_peak_log1p_losses
loc: 313
tags: [code, exp053_peak_log1p_losses]
---

# heatmap_peak_losses

> exp053 heatmap losses: Pearson + grad + log1p peak stack (hotspot / max / centroid / blob).

**Source:** `experiments/exp053_peak_log1p_losses/codes/heatmap_peak_losses.py` · 313 lines
**Experiment:** [[exp053_peak_log1p_losses]]

## Purpose

```text
exp053 heatmap losses: Pearson + grad + log1p peak stack (hotspot / max / centroid / blob).

Phys blob runs in robust log1p space (not denormed Ω). No FG z-Huber or percentile/dynrange terms.
```

## Functions

- **`_sobel_kernels(device: torch.device, dtype: torch.dtype)`**
- **`_sobel_grad(hm: torch.Tensor)`**
- **`_masked_mean(per_px: torch.Tensor, mask: torch.Tensor)`**
- **`grad_vector_field_loss(recon: torch.Tensor, target: torch.Tensor, fg: torch.Tensor, c: _tr.Config)`** — FG-masked grad loss: vector huber + flow direction.
- **`_robust_log1p_maps(recon: torch.Tensor, target: torch.Tensor, c: _tr.Config, *, pi_freq: torch.Tensor | None)`** — Map z-score heatmaps to log1p(Ω) per MHz bin (linear; no exp blow-up).
- **`_fg_percentile(flat: torch.Tensor, q: float, dim: int=1)`** — Per-row quantile on flattened FG values.
- **`hotspot_log1p_loss(recon_lp: torch.Tensor, target_lp: torch.Tensor, fg: torch.Tensor, target_z: torch.Tensor, bg: float)`** — Intensity-squared weighted L1 in log1p — focuses on bright peaks.
- **`max_log1p_loss(recon_lp: torch.Tensor, target_lp: torch.Tensor, fg: torch.Tensor)`** — |max log1p(recon) - max log1p(target)| on FG.
- **`p95_log1p_loss(recon_lp: torch.Tensor, target_lp: torch.Tensor, fg: torch.Tensor)`** — |p95 log1p(recon) - p95 log1p(target)| on FG pixels.
- **`centroid_top_pct_log1p_loss(recon_lp: torch.Tensor, target_lp: torch.Tensor, fg: torch.Tensor, *, top_q: float=0.9)`** — L2 distance between centroids of top target-percentile FG pixels in log1p.
- **`local_max_log1p_loss(recon_lp: torch.Tensor, target_lp: torch.Tensor)`** — L1 on 3x3 max-pooled log1p maps — peak neighborhood alignment.
- **`_topk_mask(intens: torch.Tensor, fg: torch.Tensor, k: int)`**
- **`_blob_under_over(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, *, undershoot_weight: float, overshoot_weight: float)`**
- **`heatmap_peak_log1p_bundle(recon: torch.Tensor, target: torch.Tensor, c: _tr.Config, *, pi_freq: torch.Tensor | None=None)`** — Weighted log1p peak losses — each value per-sample (B,).
- **`heatmap_peak_log1p_total(recon: torch.Tensor, target: torch.Tensor, c: _tr.Config, *, pi_freq: torch.Tensor | None=None)`** — Scalar-weighted sum of log1p peak terms; returns (per_sample, raw_terms).
- **`heatmap_loss_pearson_grad(recon: torch.Tensor, target: torch.Tensor, c: _tr.Config, *, lite: bool=False)`** — FG Pearson (global) + grad field (local) — per-sample (B,).
- **`heatmap_peak_blob_phys_loss(recon: torch.Tensor, target: torch.Tensor, c: _tr.Config, *, pi_freq: torch.Tensor | None=None, reduction: str='mean')`** — Top-k target blob in log1p: mask = top-k(target) only (no recon union).

## Imports

- [[experiments.exp038_true_multi.codes.train_vae_simple]]
- [[experiments.exp053_peak_log1p_losses.codes.spatial_metrics]]
- [[norm_stats]]

## Imported by

- [[experiments.exp053_peak_log1p_losses.codes.diagnose_nan_grad]]
- [[experiments.exp053_peak_log1p_losses.codes.run_epoch_encode]]
- [[experiments.exp053_peak_log1p_losses.codes.train_vae_simple]]

## External dependencies

`src_vae`, `torch`
