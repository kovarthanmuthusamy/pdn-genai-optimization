---
title: run_multitype_sim_pipeline
type: code
path: pipelines/dataset_sim/run_multitype_sim_pipeline.py
group: pipelines/dataset_sim
loc: 320
tags: [code, pipelines, runnable, uncommitted]
---

# run_multitype_sim_pipeline

> Multi-type combinations CSV -> ECADStar simulation pipeline.

**Source:** `pipelines/dataset_sim/run_multitype_sim_pipeline.py` · 320 lines
**Git:** uncommitted — not yet tracked
**Runnable:** CONFIG-only script — edit constants at top, then `python pipelines/dataset_sim/run_multitype_sim_pipeline.py`

## Purpose

```text
Multi-type combinations CSV -> ECADStar simulation pipeline.

Identical workflow to ``run_combinations_sim_pipeline.py`` but the input CSV
carries decap *type codes* (0 empty / 1 type-1 / 2 type-2). The generated PEB
sets per-component C / ESR / ESL from the type catalog so ECADStar knows which
part is placed on each slot.

Type catalog (SI units, see scrap/generation/generate_peb_multitype.py):
    Type1 : 100 nF, ESL 222 pH, ESR 8.9 mΩ
    Type2 :  47 nF, ESL 154 pH, ESR 21.4 mΩ

Workflow:
  1. ``multitype_impedance.peb`` (all rows) → simulate → ``Impedance/PI-1 … PI-N``
  2. Per MHz: ``multitype_dist_{MHz}MHz.peb`` → simulate → ``heatmaps_{MHz}MHz/PI-1 … PI-N``

Run:
    python pipelines/dataset_sim/run_multitype_sim_pipeline.py
```

## Constants

| Name | Value |
|------|-------|
| `_REPO_BOOT` | `Path(__file__).resolve().parents[2]` |
| `COMBINATIONS_CSV` | `REPO_ROOT / 'data' / 'heatmaps' / 'combinations_multitype_combined_15000.csv'` |
| `START_LAYOUT` | `0` |
| `USE_ANCHOR_MHZ` | `False` |
| `POWERBUS` | `'Power_GND'` |
| `IMPEDANCE_COMPONENTS` | `'IC1_Port1'` |
| `ECADSTAR_ERF_PATH` | `'C:\\Users\\muthusamy\\Desktop\\design\\H-shape.emc\\H-shape.erf'` |
| `ECADSTAR_EMC_OUTPUT_DIR` | `'C:\\Users\\muthusamy\\Desktop\\design\\H-shape.emc'` |
| `ENGINEER_EXE` | `'C:\\Program Files\\eCADSTAR\\eCADSTAR 2023.0\\Analysis\\bin\\engineer.exe'` |
| `ECADSTAR_CLEAR_LOCK_FILE` | `True` |
| `ECADSTAR_WAIT_TIMEOUT_SEC` | `172800` |
| `ECADSTAR_WAIT_POLL_SEC` | `120` |
| `SKIP_IMPEDANCE` | `False` |
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

- [[combinations_multitype]]
- [[move_outputs]]
- [[multifreq_anchors]]
- [[peb_multitype]]
- [[pipelines.dataset_sim.ecadstar]]
- [[pipelines.dataset_sim.paths]]
- [[repo_paths]]
- [[trigger_append]]

## External dependencies

`repo_paths`, `src_vae`
