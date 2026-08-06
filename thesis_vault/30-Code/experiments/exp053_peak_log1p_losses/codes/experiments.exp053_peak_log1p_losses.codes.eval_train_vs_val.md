---
title: eval_train_vs_val
type: code
path: experiments/exp053_peak_log1p_losses/codes/eval_train_vs_val.py
group: experiments/exp053_peak_log1p_losses/codes
experiment: exp053_peak_log1p_losses
loc: 96
tags: [code, exp053_peak_log1p_losses]
---

# eval_train_vs_val

> Diagnostic: compare layout_cross vs encode_cross on TRAIN vs VAL designs.

**Source:** `experiments/exp053_peak_log1p_losses/codes/eval_train_vs_val.py` · 96 lines
**Experiment:** [[exp053_peak_log1p_losses]]

## Purpose

```text
Diagnostic: compare layout_cross vs encode_cross on TRAIN vs VAL designs.

Answers: is poor layout generation a generalization problem (train >> val)
or a capacity problem (train ~= val)?
```

## Constants

| Name | Value |
|------|-------|
| `_ROOT` | `Path(__file__).resolve().parents[3]` |
| `EXP_DIR` | `Path(__file__).resolve().parents[1]` |

## Functions

- **`_load_cfg()`**
- **`main()`**

## Imports

- [[exp053_eval_common]]
- [[experiments.exp038_true_multi.codes.dataloader_multifreq]]
- [[experiments.exp053_peak_log1p_losses.codes.eval_spatial_metrics]]

## External dependencies

`torch`
