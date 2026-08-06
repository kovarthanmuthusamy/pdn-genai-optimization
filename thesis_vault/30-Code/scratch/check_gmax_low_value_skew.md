---
title: check_gmax_low_value_skew
type: code
path: scratch/check_gmax_low_value_skew.py
group: scratch
loc: 226
tags: [code, scratch]
---

# check_gmax_low_value_skew

> Temp check: low-value skew in data_multifreq_gmax normalized heatmaps.

**Source:** `scratch/check_gmax_low_value_skew.py` · 226 lines

## Purpose

```text
Temp check: low-value skew in data_multifreq_gmax normalized heatmaps.

Run:
    .venv/bin/python scratch/check_gmax_low_value_skew.py
```

## Constants

| Name | Value |
|------|-------|
| `DATA_DIR` | `REPO_ROOT / 'datasets/data_multifreq_gmax'` |
| `OUT_JSON` | `REPO_ROOT / 'scratch/gmax_low_value_skew_report.json'` |
| `SAMPLE_PER_MHZ` | `120` |
| `SEED` | `42` |
| `LOW_BINS` | `[0.0, 0.0005, 0.001, 0.0016, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 1.02]` |

## Functions

- **`_skew(x: np.ndarray)`**
- **`_pct(x: np.ndarray, q: float)`**
- **`main()`**

## Imports

- [[repo_paths]]

## External dependencies

`numpy`, `repo_paths`
