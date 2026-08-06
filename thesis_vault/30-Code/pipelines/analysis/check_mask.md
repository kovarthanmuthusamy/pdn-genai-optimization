---
title: check_mask
type: code
path: pipelines/analysis/check_mask.py
group: pipelines/analysis
loc: 42
tags: [code, pipelines, runnable]
---

# check_mask

> Binary PCB mask shape inspector.

**Source:** `pipelines/analysis/check_mask.py` · 42 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python pipelines/analysis/check_mask.py`

## Purpose

```text
Binary PCB mask shape inspector.

Run: python pipelines/analysis/check_mask.py
```

## Constants

| Name | Value |
|------|-------|
| `MASK_FILE` | `repo_path('configs', 'binary_mask.npy')` |
| `OUTPUT_PATH` | `Path('mask_debug.png')` |

## Functions

- **`main()`**

## Imports

- [[repo_paths]]

## External dependencies

`matplotlib`, `numpy`, `repo_paths`
