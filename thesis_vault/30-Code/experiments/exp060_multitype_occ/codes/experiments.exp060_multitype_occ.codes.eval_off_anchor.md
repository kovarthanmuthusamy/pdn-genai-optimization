---
title: eval_off_anchor
type: code
path: experiments/exp060_multitype_occ/codes/eval_off_anchor.py
group: experiments/exp060_multitype_occ/codes
experiment: exp060_multitype_occ
loc: 91
tags: [code, exp060_multitype_occ, uncommitted]
---

# eval_off_anchor

> Off-anchor eval hook for exp055 training checkpoints.

**Source:** `experiments/exp060_multitype_occ/codes/eval_off_anchor.py` · 91 lines
**Git:** uncommitted — not yet tracked
**Experiment:** [[exp060_multitype_occ]]

## Functions

- **`set_off_anchor_config(c)`**
- **`_interval(c)`**
- **`should_run_off_anchor(epoch: int, c)`**
- **`run_off_anchor_eval(model, val_loader, *, bg: float, off_anchor_mhz=(100.0, 270.0, 400.0), max_batches: int=12, device: str='cuda', out_csv=None)`**
- **`off_anchor_aggregate_score(rows: list[dict], *, kind: str='layout_cross', weights: dict | None=None)`** — Weighted mean FG MSE across off-anchor MHz (lower = better heatmap fit).

## Imports

- [[experiments.exp060_multitype_occ.codes.eval_spatial_metrics]]

## Imported by

- [[experiments.exp060_multitype_occ.codes.train_core]]
- [[experiments.exp060_multitype_occ.codes.train_vae_simple]]
