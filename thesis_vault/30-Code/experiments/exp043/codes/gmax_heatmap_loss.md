---
title: gmax_heatmap_loss
type: code
path: experiments/exp043/codes/gmax_heatmap_loss.py
group: experiments/exp043/codes
experiment: exp043
loc: 323
tags: [code, exp043]
---

# gmax_heatmap_loss

> Global-max heatmap losses for exp043 (linear [0,1] norm space).

**Source:** `experiments/exp043/codes/gmax_heatmap_loss.py` · 323 lines
**Experiment:** [[exp043]]

## Purpose

```text
Global-max heatmap losses for exp043 (linear [0,1] norm space).

Replaces log-z-scaled huber deltas and adds symmetric dynrange penalties.
```

## Functions

- **`_fg_bg_masks(target: torch.Tensor, c)`**
- **`_fg_minmax_norm(flat: torch.Tensor, fg: torch.Tensor)`** — Per-sample min-max on foreground pixels (B, N).
- **`_fg_pattern_correlation_loss(flat_r: torch.Tensor, flat_t: torch.Tensor, fg: torch.Tensor)`** — (1 - Pearson r)² on min-max normalized FG maps — targets spatial morphology.
- **`_fg_spread_loss(recon: torch.Tensor, target: torch.Tensor, fg: torch.Tensor, flat_r: torch.Tensor, flat_t: torch.Tensor, percentile_fn)`** — Penalize sparse fields and magnitude overshoot (edge spikes / runaway max).
- **`_fg_hotspot_loss(huber: torch.Tensor, target: torch.Tensor, fg: torch.Tensor, bg: float)`** — Intensity-squared weighted huber — focuses on localized high-Ω peaks.
- **`_fg_peak_centroid_loss(recon: torch.Tensor, target: torch.Tensor, fg: torch.Tensor, percentile_fn, *, top_q: float=0.92)`** — L2 distance between centroids of top target-percentile FG pixels.
- **`_fg_quantile_inverse_weights(target: torch.Tensor, fg: torch.Tensor, *, n_bins: int, power: float)`** — Batched FG weights: rare target quantile bins get higher weight (GPU vectorized).
- **`_fg_huber_base(huber: torch.Tensor, fg: torch.Tensor, target: torch.Tensor, c)`** — FG huber mean; optional quantile inverse blend to upweight rare high bins.
- **`heatmap_loss_gmax(recon: torch.Tensor, target: torch.Tensor, c, ps: float, *, dynrange_weight: float | None=None, lite: bool=False, clip_fn=None, lap_k_fn=None, percentile_fn=None)`** — Per-sample heatmap loss (B,) tuned for Ω/global_max normalization.
- **`cross_freq_heatmap_loss_gmax(model, z: torch.Tensor, K: torch.Tensor, pi_alt: torch.Tensor, hm_alt: torch.Tensor, c, ps: float, *, dynrange_weight: float | None=None, occupancy: torch.Tensor | None=None, heatmap_skips: dict[str, torch.Tensor] | None=None)`** — Decode same z at alternate PI_freq; full gmax heatmap loss by default.

## Imports

- [[experiments.exp043.codes.__init__]]
- [[experiments.exp043.codes.train_vae_simple]]
- [[heatmap_gmax_norm]]

## Imported by

- [[_bench_train_step]]
- [[_breakdown_hm_loss]]
- [[_breakdown_hm_real]]
- [[_smoke_exp043_struct]]
- [[experiments.exp043.codes.train_vae_simple]]
- [[gmax_training_patch]]

## External dependencies

`src_vae`, `torch`
