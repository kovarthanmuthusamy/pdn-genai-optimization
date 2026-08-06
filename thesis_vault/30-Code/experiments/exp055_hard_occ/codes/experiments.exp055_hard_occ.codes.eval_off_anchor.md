---
title: eval_off_anchor
type: code
path: experiments/exp055_hard_occ/codes/eval_off_anchor.py
group: experiments/exp055_hard_occ/codes
experiment: exp055_hard_occ
loc: 70
tags: [code, exp055_hard_occ]
---

# eval_off_anchor

> Off-anchor eval hook for exp055 training checkpoints.

**Source:** `experiments/exp055_hard_occ/codes/eval_off_anchor.py` · 70 lines
**Experiment:** [[exp055_hard_occ]]

## Functions

- **`set_off_anchor_config(c)`**
- **`_interval(c)`**
- **`should_run_off_anchor(epoch: int, c)`**
- **`run_off_anchor_eval(model, val_loader, *, bg: float, off_anchor_mhz=(100.0, 270.0, 400.0), max_batches: int=12, device: str='cuda', out_csv=None)`**

## Imports

- [[experiments.exp055_hard_occ.codes.eval_spatial_metrics]]

## Imported by

- [[experiments.exp055_hard_occ.codes.train_core]]
- [[experiments.exp055_hard_occ.codes.train_vae_simple]]
