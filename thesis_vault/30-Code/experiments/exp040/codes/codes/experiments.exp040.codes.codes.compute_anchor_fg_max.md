---
title: compute_anchor_fg_max
type: code
path: experiments/exp040/codes/codes/compute_anchor_fg_max.py
group: experiments/exp040/codes/codes
experiment: exp040
loc: 70
tags: [code, exp040]
---

# compute_anchor_fg_max

> Compute per-anchor foreground max (physical Ω) from data_multifreq_norm and write metrics JSON.

**Source:** `experiments/exp040/codes/codes/compute_anchor_fg_max.py` · 70 lines
**Experiment:** [[exp040]]

## Constants

| Name | Value |
|------|-------|
| `ROOT` | `Path(__file__).resolve().parents[3]` |
| `DATA` | `ROOT / 'datasets' / 'data_multifreq_norm'` |
| `OUT` | `Path(__file__).resolve().parents[1] / 'metrics' / 'anchor_hm_fg_max.json'` |

## Functions

- **`main()`**

## Imports

- [[experiments.exp038_true_multi.codes.dataloader_multifreq]]

## External dependencies

`numpy`
