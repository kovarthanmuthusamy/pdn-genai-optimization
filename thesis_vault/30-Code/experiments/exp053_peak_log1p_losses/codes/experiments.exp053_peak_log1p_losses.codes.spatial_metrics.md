---
title: spatial_metrics
type: code
path: experiments/exp053_peak_log1p_losses/codes/spatial_metrics.py
group: experiments/exp053_peak_log1p_losses/codes
experiment: exp053_peak_log1p_losses
loc: 125
tags: [code, exp053_peak_log1p_losses]
---

# spatial_metrics

> Foreground spatial metrics — Pearson correlation and soft peak location.

**Source:** `experiments/exp053_peak_log1p_losses/codes/spatial_metrics.py` · 125 lines
**Experiment:** [[exp053_peak_log1p_losses]]

## Functions

- **`pearson_fg(recon: torch.Tensor, target: torch.Tensor, bg: float, *, margin: float=0.5, eps: float=1e-06)`** — Per-sample Pearson r on foreground pixels; returns (B,) in [-1, 1].
- **`pearson_fg_loss(recon: torch.Tensor, target: torch.Tensor, fg: torch.Tensor, *, eps: float=0.0001)`** — Per-sample loss 1 - Pearson r on fg-masked maps; returns (B,).
- **`soft_peak_coords(hm: torch.Tensor, fg: torch.Tensor, bg: float)`** — Soft-argmax peak (y, x) normalized to [0, 1]; returns (B,), (B,).
- **`peak_loc_err(recon: torch.Tensor, target: torch.Tensor, bg: float, *, margin: float=0.5)`** — L2 distance between soft peak coords (normalized); returns (B,).
- **`peak_loc_loss(recon: torch.Tensor, target: torch.Tensor, fg: torch.Tensor, bg: float)`** — Squared peak-coordinate error; returns (B,).
- **`peak_loc_sharp_loss(recon: torch.Tensor, target: torch.Tensor, fg: torch.Tensor, bg: float, *, temperature: float=0.05)`** — Low-temperature soft-argmax peak match (closer to hard argmax than peak_loc_loss).

## Imported by

- [[experiments.exp053_peak_log1p_losses.codes.eval_real_data_sweep]]
- [[experiments.exp053_peak_log1p_losses.codes.eval_spatial_metrics]]
- [[experiments.exp053_peak_log1p_losses.codes.heatmap_peak_losses]]
- [[experiments.exp053_peak_log1p_losses.codes.train_vae_simple]]

## External dependencies

`torch`
