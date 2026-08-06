---
title: clean_train_metadata
type: code
path: pipelines/dataset/clean_train_metadata.py
group: pipelines/dataset
loc: 116
tags: [code, pipelines]
---

# clean_train_metadata

> Remove append progress, batch registries, and manifest backups from data_multifreq_train.

**Source:** `pipelines/dataset/clean_train_metadata.py` · 116 lines

## Purpose

```text
Remove append progress, batch registries, and manifest backups from data_multifreq_train.

Keeps only training essentials:
  manifest.csv, dataset_meta.json, heatmap/, PI_freq/, layouts/
  (+ runtime caches multifreq_meta.json, k_values_cache.npy if present)

Run:
    python pipelines/dataset/clean_train_metadata.py
    python pipelines/dataset/clean_train_metadata.py --execute
```

## Constants

| Name | Value |
|------|-------|
| `_REPO` | `Path(__file__).resolve().parents[2]` |
| `TRAIN_ROOT` | `REPO_ROOT / 'datasets' / 'data_multifreq_train'` |
| `REMOVE_FILES` | `('append_combinations_progress.json', 'append_merged_progress.json', 'combinations_append…` |
| `REMOVE_DIRS` | `('append_batches',)` |
| `KEEP_FILES` | `frozenset({'manifest.csv', 'dataset_meta.json', 'multifreq_meta.json', 'k_values_cache.np…` |

## Functions

- **`clean_train_metadata(root: Path, *, execute: bool)`**
- **`main()`**

## Imports

- [[repo_paths]]

## External dependencies

`repo_paths`
