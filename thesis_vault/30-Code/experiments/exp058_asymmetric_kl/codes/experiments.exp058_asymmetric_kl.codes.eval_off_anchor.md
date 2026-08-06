---
title: eval_off_anchor
type: code
path: experiments/exp058_asymmetric_kl/codes/eval_off_anchor.py
group: experiments/exp058_asymmetric_kl/codes
experiment: exp058_asymmetric_kl
loc: 93
tags: [code, exp058_asymmetric_kl, uncommitted]
---

# eval_off_anchor

> Off-anchor eval hook for exp055 training checkpoints.

**Source:** `experiments/exp058_asymmetric_kl/codes/eval_off_anchor.py` · 93 lines
**Git:** uncommitted — not yet tracked
**Experiment:** [[exp058_asymmetric_kl]]

## Functions

- **`set_off_anchor_config(c)`**
- **`_interval(c)`**
- **`should_run_off_anchor(epoch: int, c)`**
- **`run_off_anchor_eval(model, val_loader, *, bg: float, off_anchor_mhz=(100.0, 270.0, 400.0), max_batches: int=12, device: str='cuda', out_csv=None)`**
- **`off_anchor_aggregate_score(rows: list[dict], *, kind: str='layout_cross', weights: dict | None=None)`** — Weighted mean FG MSE across off-anchor MHz (lower = better heatmap fit).

## Imports

- [[experiments.exp058_asymmetric_kl.codes.eval_spatial_metrics]]

## Imported by

- [[experiments.exp058_asymmetric_kl.codes.train_core]]
- [[experiments.exp058_asymmetric_kl.codes.train_vae_simple]]
