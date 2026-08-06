---
title: spatial_metrics
type: code
path: experiments/exp052_unbounded_pearson/codes/spatial_metrics.py
group: experiments/exp052_unbounded_pearson/codes
experiment: exp052_unbounded_pearson
loc: 96
tags: [code, exp052_unbounded_pearson]
---

# spatial_metrics

> Foreground spatial metrics — Pearson correlation and soft peak location.

**Source:** `experiments/exp052_unbounded_pearson/codes/spatial_metrics.py` · 96 lines
**Experiment:** [[exp052_unbounded_pearson]]

## Functions

- **`pearson_fg(recon: torch.Tensor, target: torch.Tensor, bg: float, *, margin: float=0.5, eps: float=1e-06)`** — Per-sample Pearson r on foreground pixels; returns (B,) in [-1, 1].
- **`pearson_fg_loss(recon: torch.Tensor, target: torch.Tensor, fg: torch.Tensor, *, eps: float=0.0001)`** — Per-sample loss 1 - Pearson r on fg-masked maps; returns (B,).
- **`soft_peak_coords(hm: torch.Tensor, fg: torch.Tensor, bg: float)`** — Soft-argmax peak (y, x) normalized to [0, 1]; returns (B,), (B,).
- **`peak_loc_err(recon: torch.Tensor, target: torch.Tensor, bg: float, *, margin: float=0.5)`** — L2 distance between soft peak coords (normalized); returns (B,).
- **`peak_loc_loss(recon: torch.Tensor, target: torch.Tensor, fg: torch.Tensor, bg: float)`** — Squared peak-coordinate error; returns (B,).

## Imported by

- [[experiments.exp052_unbounded_pearson.codes.eval_real_data_sweep]]
- [[experiments.exp052_unbounded_pearson.codes.eval_spatial_metrics]]
- [[experiments.exp052_unbounded_pearson.codes.heatmap_peak_losses]]
- [[experiments.exp052_unbounded_pearson.codes.run_epoch_encode]]
- [[experiments.exp052_unbounded_pearson.codes.train_vae_simple]]

## External dependencies

`torch`
