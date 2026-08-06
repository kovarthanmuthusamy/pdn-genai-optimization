---
title: unbounded_heatmap_loss
type: code
path: experiments/exp045/codes/unbounded_heatmap_loss.py
group: experiments/exp045/codes
experiment: exp045
loc: 140
tags: [code, exp045]
---

# unbounded_heatmap_loss

> Robust unbounded heatmap loss — tail-weighted Huber, no recon clip (exp045).

**Source:** `experiments/exp045/codes/unbounded_heatmap_loss.py` · 140 lines
**Experiment:** [[exp045]]

## Functions

- **`_tail_weights(target: torch.Tensor, bg: float, thr: float, boost: float)`** — Upweight high-z foreground pixels (rare peaks).
- **`heatmap_loss_unbounded(recon: torch.Tensor, target: torch.Tensor, c: _tr.Config, ps: float, *, dynrange_weight: float | None=None, lite: bool=False)`** — Per-sample heatmap loss (B,) — no recon z-clip; robust Huber + tail weights.
- **`_heatmap_loss_body(recon: torch.Tensor, target: torch.Tensor, c: _tr.Config, ps: float, *, dynrange_weight: float | None=None, lite: bool=False)`** — Full heatmap loss without recon clip.
- **`heatmap_phys_amplitude_loss_unbounded(recon: torch.Tensor, target: torch.Tensor, hm_log_mean: float, hm_log_std: float, c: _tr.Config)`** — Physical p99 loss — soft clamp at z_max instead of hard clip.
- **`patch_unbounded_losses()`** — Replace heatmap losses in exp038 trainer with unbounded variants.

## Imports

- [[experiments.exp038_true_multi.codes.train_vae_simple]]
- [[experiments.exp045.codes.spatial_metrics]]

## Imported by

- [[experiments.exp045.codes.train_vae_simple]]

## External dependencies

`torch`
