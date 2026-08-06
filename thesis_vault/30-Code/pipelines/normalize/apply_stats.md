---
title: apply_stats
type: code
path: pipelines/normalize/apply_stats.py
group: pipelines/normalize
loc: 236
tags: [code, pipelines, runnable]
---

# apply_stats

> Normalize a held-out dataset using training normalization stats.

**Source:** `pipelines/normalize/apply_stats.py` · 236 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python pipelines/normalize/apply_stats.py`

## Purpose

```text
Normalize a held-out dataset using training normalization stats.

Run: python pipelines/normalize/apply_stats.py
```

## Constants

| Name | Value |
|------|-------|
| `IN_ROOT` | `Path('datasets/data_eval')` |
| `OUT_ROOT` | `Path('datasets/data_eval_norm')` |
| `STATS_JSON` | `Path('datasets/data_norm/normalization_stats.json')` |
| `OVERWRITE` | `False` |
| `PRINT_EVERY` | `2000` |
| `_SAMPLE_RE` | `re.compile('sample_(\\d+)\\.npy$')` |

## Functions

- **`_parse_sample_id(name: str)`**
- **`_load_stats(stats_json: Path)`**
- **`_normalize_heatmap(raw: np.ndarray, *, hm_stats: dict)`**
- **`_normalize_impedance(raw: np.ndarray, *, imp_stats: dict)`**
- **`main()`**

## Imports

- [[dataset_meta]]
- [[repo_paths]]

## External dependencies

`libs`, `numpy`, `repo_paths`
