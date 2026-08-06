---
title: eval_off_anchor
type: code
path: experiments/exp054_K_30/codes/eval_off_anchor.py
group: experiments/exp054_K_30/codes
experiment: exp054_K_30
loc: 68
tags: [code, exp054_K_30]
---

# eval_off_anchor

> Off-anchor eval hook for exp054 training checkpoints.

**Source:** `experiments/exp054_K_30/codes/eval_off_anchor.py` · 68 lines
**Experiment:** [[exp054_K_30]]

## Functions

- **`set_off_anchor_config(c)`**
- **`_interval(c)`**
- **`should_run_off_anchor(epoch: int, c)`**
- **`run_off_anchor_eval(model, val_loader, *, bg: float, off_anchor_mhz=(100.0, 270.0, 400.0), max_batches: int=12, device: str='cuda', out_csv=None)`**

## Imports

- [[experiments.exp054_K_30.codes.eval_spatial_metrics]]

## Imported by

- [[experiments.exp054_K_30.codes.train_core]]
- [[experiments.exp054_K_30.codes.train_vae_simple]]
