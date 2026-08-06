---
title: eval_train_vs_val
type: code
path: experiments/exp047/codes/eval_train_vs_val.py
group: experiments/exp047/codes
experiment: exp047
loc: 96
tags: [code, exp047]
---

# eval_train_vs_val

> Diagnostic: compare layout_cross vs encode_cross on TRAIN vs VAL designs.

**Source:** `experiments/exp047/codes/eval_train_vs_val.py` · 96 lines
**Experiment:** [[exp047]]

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

- [[exp047_eval_common]]
- [[experiments.exp038_true_multi.codes.dataloader_multifreq]]
- [[experiments.exp047.codes.eval_spatial_metrics]]

## External dependencies

`torch`
