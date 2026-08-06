---
title: heatmap_peak_losses
type: code
path: experiments/exp060_multitype_occ/codes/heatmap_peak_losses.py
group: experiments/exp060_multitype_occ/codes
experiment: exp060_multitype_occ
loc: 208
tags: [code, exp060_multitype_occ, uncommitted]
---

# heatmap_peak_losses

> Heatmap losses: Pearson+grad tier_a + log1p peak/valley extrema.

**Source:** `experiments/exp060_multitype_occ/codes/heatmap_peak_losses.py` · 208 lines
**Git:** uncommitted — not yet tracked
**Experiment:** [[exp060_multitype_occ]]

## Functions

- **`_sobel(hm)`**
- **`_finite(t, *, posinf=10000.0, neginf=0.0)`**
- **`grad_vector_field_loss(recon, target, fg, c)`**
- **`_robust_log1p_maps(recon, target, c, *, pi_freq)`**
- **`_prep_log1p_fg(recon, target, c, *, pi_freq)`**
- **`_fg_percentile(flat, q, dim=1)`**
- **`_salience_log1p_l1(recon_lp, tgt_lp, salience)`**
- **`_fg_extrema_log1p(recon_lp, tgt_lp, fg, *, side)`**
- **`_centroid_log1p(recon_lp, tgt_lp, fg, *, side, c)`**
- **`_topregion_huber(recon_lp, tgt_lp, fg, *, side, c)`** — Huber on log1p amplitude, restricted to top (peak) / bottom (valley) GT percentile FG pixels.
- **`_spot_salience(target_z, fg, bg, *, side)`**
- **`_term(kind, recon_lp, tgt_lp, fg, target_z, bg, *, side, c)`**
- **`_bundle(recon_lp, tgt_lp, fg, target_z, bg, c, *, side)`**
- **`_total(terms, c, *, side, recon)`**
- **`heatmap_extrema_log1p_both(recon, target, c, *, pi_freq=None, sides=('peak', 'valley'))`**
- **`heatmap_loss_pearson_grad(recon, target, c, *, lite=False)`**

## Imports

- [[experiments.exp060_multitype_occ.codes.spatial_metrics]]
- [[norm_stats]]

## Imported by

- [[experiments.exp060_multitype_occ.codes.train_vae_simple]]

## External dependencies

`src_vae`, `torch`
