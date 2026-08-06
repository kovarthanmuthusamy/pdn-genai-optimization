---
title: spatial_metrics
type: code
path: experiments/exp060_multitype_occ/codes/spatial_metrics.py
group: experiments/exp060_multitype_occ/codes
experiment: exp060_multitype_occ
loc: 73
tags: [code, exp060_multitype_occ, uncommitted]
---

# spatial_metrics

> FG Pearson + soft extrema location metrics.

**Source:** `experiments/exp060_multitype_occ/codes/spatial_metrics.py` · 73 lines
**Git:** uncommitted — not yet tracked
**Experiment:** [[exp060_multitype_occ]]

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

- [[experiments.exp060_multitype_occ.codes.eval_spatial_metrics]]
- [[experiments.exp060_multitype_occ.codes.heatmap_peak_losses]]
- [[experiments.exp060_multitype_occ.codes.run_epoch_encode]]
- [[experiments.exp060_multitype_occ.codes.train_vae_simple]]

## External dependencies

`torch`
