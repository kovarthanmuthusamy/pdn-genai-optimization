---
title: eval_off_anchor
type: code
path: experiments/exp056_graph_vae/codes/eval_off_anchor.py
group: experiments/exp056_graph_vae/codes
experiment: exp056_graph_vae
loc: 70
tags: [code, exp056_graph_vae]
---

# eval_off_anchor

> Off-anchor eval hook for exp055 training checkpoints.

**Source:** `experiments/exp056_graph_vae/codes/eval_off_anchor.py` · 70 lines
**Experiment:** [[exp056_graph_vae]]

## Functions

- **`set_off_anchor_config(c)`**
- **`_interval(c)`**
- **`should_run_off_anchor(epoch: int, c)`**
- **`run_off_anchor_eval(model, val_loader, *, bg: float, off_anchor_mhz=(100.0, 270.0, 400.0), max_batches: int=12, device: str='cuda', out_csv=None)`**

## Imports

- [[experiments.exp056_graph_vae.codes.eval_spatial_metrics]]

## Imported by

- [[experiments.exp056_graph_vae.codes.train_core]]
- [[experiments.exp056_graph_vae.codes.train_vae_simple]]
