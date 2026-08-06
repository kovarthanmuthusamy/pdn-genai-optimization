---
title: eval_cross_freq_gmax
type: code
path: experiments/exp043/codes/eval_cross_freq_gmax.py
group: experiments/exp043/codes
experiment: exp043
loc: 96
tags: [code, exp043]
---

# eval_cross_freq_gmax

> Off-anchor eval for exp043 global-max + optional log1p train space.

**Source:** `experiments/exp043/codes/eval_cross_freq_gmax.py` · 96 lines
**Experiment:** [[exp043]]

## Functions

- **`_fg_mse(recon: torch.Tensor, target: torch.Tensor, fg_thr: float)`**
- **`run_off_anchor_eval_gmax(model: torch.nn.Module, val_loader, *, c, off_anchor_mhz: tuple[float, ...], max_batches: int=30, device: str | torch.device='cuda', out_csv: Path | str | None=None)`** — Layout-z / encode-cross at off-anchor MHz; targets in train space.

## Imports

- [[gmax_training_patch]]
- [[heatmap_gmax_norm]]
- [[pi_freq_utils]]

## Imported by

- [[experiments.exp043.codes.train_vae_simple]]

## External dependencies

`src_vae`, `torch`
