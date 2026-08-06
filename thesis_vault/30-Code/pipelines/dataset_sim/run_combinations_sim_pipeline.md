---
title: run_combinations_sim_pipeline
type: code
path: pipelines/dataset_sim/run_combinations_sim_pipeline.py
group: pipelines/dataset_sim
loc: 282
tags: [code, pipelines, runnable]
---

# run_combinations_sim_pipeline

> Combinations.csv → ECADStar simulation pipeline (impedance, then PI-Distribution per MHz).

**Source:** `pipelines/dataset_sim/run_combinations_sim_pipeline.py` · 282 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python pipelines/dataset_sim/run_combinations_sim_pipeline.py`

## Purpose

```text
Combinations.csv → ECADStar simulation pipeline (impedance, then PI-Distribution per MHz).

One .peb file per phase containing **all** layouts in the selected CSV slice (default 10,000).
ECADStar creates PI-1, PI-2, … in PEB order; outputs are moved with those names unchanged.

Workflow:
  1. ``combinations_impedance.peb`` (all rows) → simulate → ``Impedance/PI-1 … PI-N``
  2. Per MHz: ``combinations_dist_{MHz}MHz.peb`` → simulate → ``heatmaps_{MHz}MHz/PI-1 … PI-N``

Reference: ``scrap/orchestration/run_multifreq_sweep_pipeline.py``

Run:
    python pipelines/dataset_sim/run_combinations_sim_pipeline.py
```

## Constants

| Name | Value |
|------|-------|
| `_REPO_BOOT` | `Path(__file__).resolve().parents[2]` |
| `COMBINATIONS_CSV` | `REPO_ROOT / 'data' / 'heatmaps' / 'peb_with_29k' / 'all_combinations_merged.csv'` |
| `START_LAYOUT` | `0` |
| `USE_ANCHOR_MHZ` | `False` |
| `POWERBUS` | `'Power_GND'` |
| `IMPEDANCE_COMPONENTS` | `'IC1_Port1,IC2_Port2'` |
| `ECADSTAR_ERF_PATH` | `'C:\\Users\\muthusamy\\Desktop\\design\\H-shape.emc\\H-shape.erf'` |
| `ECADSTAR_EMC_OUTPUT_DIR` | `'C:\\Users\\muthusamy\\Desktop\\design\\H-shape.emc'` |
| `ENGINEER_EXE` | `'C:\\Program Files\\eCADSTAR\\eCADSTAR 2023.0\\Analysis\\bin\\engineer.exe'` |
| `ECADSTAR_CLEAR_LOCK_FILE` | `True` |
| `ECADSTAR_WAIT_TIMEOUT_SEC` | `172800` |
| `ECADSTAR_WAIT_POLL_SEC` | `120` |
| `SKIP_IMPEDANCE` | `True` |
| `SKIP_DISTRIBUTION` | `False` |
| `SKIP_SIMULATE` | `False` |
| `SKIP_MOVE` | `False` |
| `CLEAN_DEST_BEFORE_PASTE` | `True` |
| `RESUME_FROM_PROGRESS` | `True` |
| `TRIGGER_APPEND_AFTER_MOVE` | `True` |
| `OUTPUT_ROOT` | `resolve_raw_root()` |
| `PEB_DIR` | `resolve_windows_path(PEB_DIR_WIN)` |

## Functions

- **`_mhz_list()`**
- **`_progress_path()`**
- **`_load_progress()`**
- **`_save_progress(progress: dict)`**
- **`_simulate_peb(peb_path: Path, *, pi_count: int, mode: str, expected_mhz: float | None=None)`**
- **`run()`**
- **`main()`**

## Imports

- [[combinations]]
- [[move_outputs]]
- [[multifreq_anchors]]
- [[peb]]
- [[pipelines.dataset_sim.ecadstar]]
- [[pipelines.dataset_sim.paths]]
- [[repo_paths]]
- [[trigger_append]]

## Imported by

- [[verify_pipeline_paths]]

## External dependencies

`repo_paths`, `src_vae`
