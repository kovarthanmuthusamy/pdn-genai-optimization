---
title: compute_stats
type: code
path: pipelines/normalize/compute_stats.py
group: pipelines/normalize
loc: 222
tags: [code, pipelines, runnable]
---

# compute_stats

> Raw dataset min/max and percentile statistics.

**Source:** `pipelines/normalize/compute_stats.py` · 222 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python pipelines/normalize/compute_stats.py`

## Purpose

```text
Raw dataset min/max and percentile statistics.

Run: python pipelines/normalize/compute_stats.py
```

## Constants

| Name | Value |
|------|-------|
| `DATA_DIR` | `_REPO_ROOT / 'source' / 'data_2'` |
| `OUTPUT_FILE` | `_REPO_ROOT / 'source' / 'data_norm' / 'normalization_stats.json'` |
| `PERCENTILE_LOWER` | `1` |
| `PERCENTILE_UPPER` | `99` |

## Functions

- **`calculate_stats(data_dir, percentile_lower=1, percentile_upper=99)`** — Calculate normalization statistics from raw data.
- **`save_stats(stats, output_file)`** — Save statistics to JSON file.
- **`display_stats_summary(stats)`** — Display a formatted summary of the calculated statistics.
- **`main()`** — Main function.

## Imports

- [[repo_paths]]

## External dependencies

`numpy`, `repo_paths`, `tqdm`
