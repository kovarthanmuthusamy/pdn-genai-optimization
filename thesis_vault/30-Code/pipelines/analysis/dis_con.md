---
title: dis_con
type: code
path: pipelines/analysis/dis_con.py
group: pipelines/analysis
loc: 51
tags: [code, pipelines, runnable]
---

# dis_con

> Sigmoid-smooth 3-channel occupancy heatmap batch processor.

**Source:** `pipelines/analysis/dis_con.py` · 51 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python pipelines/analysis/dis_con.py`

## Purpose

```text
Sigmoid-smooth 3-channel occupancy heatmap batch processor.

Run: python pipelines/analysis/dis_con.py
```

## Constants

| Name | Value |
|------|-------|
| `INPUT_DIR` | `Path('src/data_2/Occ_map')` |
| `OUTPUT_DIR` | `Path('src/data_norm/Occ_map')` |
| `SIGMOID_ALPHA` | `5.0` |

## Functions

- **`process_channels(path: Path, out: Path, *, alpha: float)`**
- **`main()`**

## Imports

- [[repo_paths]]

## External dependencies

`numpy`, `repo_paths`, `torch`
