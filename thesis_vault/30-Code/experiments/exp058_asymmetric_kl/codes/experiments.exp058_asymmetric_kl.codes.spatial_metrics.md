---
title: spatial_metrics
type: code
path: experiments/exp058_asymmetric_kl/codes/spatial_metrics.py
group: experiments/exp058_asymmetric_kl/codes
experiment: exp058_asymmetric_kl
loc: 73
tags: [code, exp058_asymmetric_kl, uncommitted]
---

# spatial_metrics

> FG Pearson + soft extrema location metrics.

**Source:** `experiments/exp058_asymmetric_kl/codes/spatial_metrics.py` · 73 lines
**Git:** uncommitted — not yet tracked
**Experiment:** [[exp058_asymmetric_kl]]

## Functions

- **`pearson_fg(recon, target, bg, *, margin=0.5, eps=1e-06)`**
- **`pearson_fg_loss(recon, target, fg, *, eps=0.0001)`**
- **`_coord_grid(b, h, w, device, dtype)`**
- **`_extrema_logits(hm, fg, bg, *, side, temperature)`**
- **`_sharp_extrema_coords(hm, fg, bg, *, side, temperature)`**
- **`soft_peak_coords(hm, fg, bg)`**
- **`peak_loc_err(recon, target, bg, *, margin=0.5)`**
- **`extrema_loc_sharp_loss(recon, target, fg, bg, *, side, temperature=0.05)`**

## Imported by

- [[experiments.exp058_asymmetric_kl.codes.eval_spatial_metrics]]
- [[experiments.exp058_asymmetric_kl.codes.heatmap_peak_losses]]
- [[experiments.exp058_asymmetric_kl.codes.train_vae_simple]]

## External dependencies

`torch`
