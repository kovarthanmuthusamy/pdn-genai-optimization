---
title: paths
type: code
path: pipelines/dataset_sim/paths.py
group: pipelines/dataset_sim
loc: 58
tags: [code, pipelines]
---

# paths

> Shared Windows paths and heatmap location helpers for sim → append pipeline.

**Source:** `pipelines/dataset_sim/paths.py` · 58 lines

## Constants

| Name | Value |
|------|-------|
| `RAW_ROOT_WIN` | `'C:\\Users\\muthusamy\\Desktop\\Raw'` |
| `PEB_DIR_WIN` | `'C:\\Users\\muthusamy\\Desktop\\Raw\\peb'` |

## Functions

- **`resolve_raw_root(win_path: str | None=None)`**
- **`heatmap_raw_dir(raw_root: Path, mhz: float)`** — Folder append reads and sim writes (``heatmaps_{MHz}MHz``, legacy ``heatmap_*`` ok).
- **`heatmap_map_path(pi_dir: Path, mhz: float)`**
- **`verify_heatmap_ready(mhz: float, *, raw_root: Path | None=None, raw_root_win: str | None=None)`** — Raise if sim move output is not ready for append at this MHz.

## Imports

- [[move_outputs]]
- [[pipelines.dataset_sim.ecadstar]]

## Imported by

- [[append_merged_combinations_multifreq]]
- [[append_restore_49k_legacy_multifreq]]
- [[ingest_labels]]
- [[run_append_locked]]
- [[run_append_worker]]
- [[run_combinations_sim_pipeline]]
- [[run_multitype_sim_pipeline]]
- [[trigger_append]]
- [[verify_pipeline_paths]]
