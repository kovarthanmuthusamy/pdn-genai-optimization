---
title: run_append_worker
type: code
path: pipelines/dataset_sim/run_append_worker.py
group: pipelines/dataset_sim
loc: 184
tags: [code, pipelines]
---

# run_append_worker

> Process append_merged queue jobs one at a time (survives sim pipeline continuing).

**Source:** `pipelines/dataset_sim/run_append_worker.py` · 184 lines

## Purpose

```text
Process append_merged queue jobs one at a time (survives sim pipeline continuing).

Run as worker (default):
    python pipelines/dataset_sim/run_append_worker.py

Enqueue recovery / manual jobs:
    python pipelines/dataset_sim/run_append_worker.py --enqueue 470 490
```

## Constants

| Name | Value |
|------|-------|
| `_REPO` | `Path(__file__).resolve().parents[2]` |
| `APPEND_SCRIPT` | `REPO_ROOT / 'pipelines' / 'data' / 'append_merged_combinations_multifreq.py'` |
| `APPEND_LOCK_FILE` | `REPO_ROOT / 'logs' / '.append_merged.lock'` |
| `OUTPUT_ROOT` | `REPO_ROOT / 'datasets' / 'data_multifreq_train'` |
| `_STOP_REQUESTED` | `False` |

## Functions

- **`_handle_stop(signum, _frame)`**
- **`_refresh_meta()`**
- **`_sanity_check(mhz_tag: int, append_tag: str)`**
- **`_run_one_job(job: dict)`**
- **`_enqueue_mhz_list(mhz_values: list[float], *, raw_root: str)`**
- **`main()`**

## Imports

- [[append_queue]]
- [[dataset_meta]]
- [[pipelines.dataset_sim.paths]]
- [[repo_paths]]
- [[trigger_append]]

## External dependencies

`fcntl`, `libs`, `repo_paths`, `signal`
