---
title: compute_anchor_fg_max
type: code
path: experiments/exp038_true_multi/codes/compute_anchor_fg_max.py
group: experiments/exp038_true_multi/codes
experiment: exp038_true_multi
loc: 70
tags: [code, exp038_true_multi]
---

# compute_anchor_fg_max

> Compute per-anchor foreground max (physical Ω) from data_multifreq_norm and write metrics JSON.

**Source:** `experiments/exp038_true_multi/codes/compute_anchor_fg_max.py` · 70 lines
**Experiment:** [[exp038_true_multi]]

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
