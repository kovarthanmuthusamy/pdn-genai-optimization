---
title: run_append_locked
type: code
path: pipelines/dataset_sim/run_append_locked.py
group: pipelines/dataset_sim
loc: 88
tags: [code, pipelines]
---

# run_append_locked

> Run one append job under an exclusive lock; refresh dataset_meta on success.

**Source:** `pipelines/dataset_sim/run_append_locked.py` · 88 lines

## Constants

| Name | Value |
|------|-------|
| `_REPO` | `Path(__file__).resolve().parents[2]` |
| `APPEND_SCRIPT` | `REPO_ROOT / 'pipelines' / 'data' / 'append_merged_combinations_multifreq.py'` |
| `LOCK_FILE` | `REPO_ROOT / 'logs' / '.append_merged.lock'` |
| `OUTPUT_ROOT` | `REPO_ROOT / 'datasets' / 'data_multifreq_train'` |

## Functions

- **`_refresh_meta()`**
- **`main()`**

## Imports

- [[dataset_meta]]
- [[pipelines.dataset_sim.paths]]
- [[repo_paths]]

## External dependencies

`fcntl`, `libs`, `repo_paths`
