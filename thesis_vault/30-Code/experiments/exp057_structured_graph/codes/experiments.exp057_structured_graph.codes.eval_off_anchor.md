---
title: eval_off_anchor
type: code
path: experiments/exp057_structured_graph/codes/eval_off_anchor.py
group: experiments/exp057_structured_graph/codes
experiment: exp057_structured_graph
loc: 93
tags: [code, exp057_structured_graph]
---

# eval_off_anchor

> Off-anchor eval hook for exp055 training checkpoints.

**Source:** `experiments/exp057_structured_graph/codes/eval_off_anchor.py` · 93 lines
**Experiment:** [[exp057_structured_graph]]

## Functions

- **`set_off_anchor_config(c)`**
- **`_interval(c)`**
- **`should_run_off_anchor(epoch: int, c)`**
- **`run_off_anchor_eval(model, val_loader, *, bg: float, off_anchor_mhz=(100.0, 270.0, 400.0), max_batches: int=12, device: str='cuda', out_csv=None)`**
- **`off_anchor_aggregate_score(rows: list[dict], *, kind: str='layout_cross', weights: dict | None=None)`** — Weighted mean FG MSE across off-anchor MHz (lower = better heatmap fit).

## Imports

- [[experiments.exp057_structured_graph.codes.eval_spatial_metrics]]

## Imported by

- [[experiments.exp057_structured_graph.codes.train_core]]
- [[experiments.exp057_structured_graph.codes.train_vae_simple]]
