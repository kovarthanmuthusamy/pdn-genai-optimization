---
title: spatial_metrics
type: code
path: experiments/exp051_new_datas_appended/codes/spatial_metrics.py
group: experiments/exp051_new_datas_appended/codes
experiment: exp051_new_datas_appended
loc: 94
tags: [code, exp051_new_datas_appended]
---

# spatial_metrics

> Foreground spatial metrics — Pearson correlation and soft peak location.

**Source:** `experiments/exp051_new_datas_appended/codes/spatial_metrics.py` · 94 lines
**Experiment:** [[exp051_new_datas_appended]]

## Functions

- **`pearson_fg(recon: torch.Tensor, target: torch.Tensor, bg: float, *, margin: float=0.5, eps: float=1e-06)`** — Per-sample Pearson r on foreground pixels; returns (B,) in [-1, 1].
- **`pearson_fg_loss(recon: torch.Tensor, target: torch.Tensor, fg: torch.Tensor, *, eps: float=1e-06)`** — Per-sample loss 1 - Pearson r on fg-masked maps; returns (B,).
- **`soft_peak_coords(hm: torch.Tensor, fg: torch.Tensor, bg: float)`** — Soft-argmax peak (y, x) normalized to [0, 1]; returns (B,), (B,).
- **`peak_loc_err(recon: torch.Tensor, target: torch.Tensor, bg: float, *, margin: float=0.5)`** — L2 distance between soft peak coords (normalized); returns (B,).
- **`peak_loc_loss(recon: torch.Tensor, target: torch.Tensor, fg: torch.Tensor, bg: float)`** — Squared peak-coordinate error; returns (B,).

## Imported by

- [[experiments.exp051_new_datas_appended.codes.eval_real_data_sweep]]
- [[experiments.exp051_new_datas_appended.codes.eval_spatial_metrics]]
- [[experiments.exp051_new_datas_appended.codes.run_epoch_encode]]
- [[experiments.exp051_new_datas_appended.codes.train_vae_simple]]

## External dependencies

`torch`
