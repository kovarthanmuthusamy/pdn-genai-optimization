---
title: subsample_inverse_k
type: code
path: pipelines/dataset/subsample_inverse_k.py
group: pipelines/dataset
loc: 370
tags: [code, pipelines, runnable]
---

# subsample_inverse_k

> Inverse-K exponential subsampling of multifreq dataset layouts.

**Source:** `pipelines/dataset/subsample_inverse_k.py` · 370 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python pipelines/dataset/subsample_inverse_k.py`

## Purpose

```text
Inverse-K exponential subsampling of multifreq dataset layouts.

Run:
    python pipelines/dataset/subsample_inverse_k.py
```

## Constants

| Name | Value |
|------|-------|
| `DATA_DIR` | `_ROOT / 'datasets' / 'data_multifreq_norm'` |
| `EXECUTE` | `False` |
| `SEED` | `42` |
| `N_REF` | `1000.0` |
| `TAU` | `12.0` |
| `TAU_LOW` | `35.0` |
| `K_LOW_MAX` | `20` |
| `K_ANCHOR` | `2` |
| `N_MIN` | `150` |
| `N_MAX` | `1000` |
| `KEEP_EDGE_K` | `False` |
| `MANIFEST_FIELDS` | `['sample_name', 'design_id', 'freq_label', 'freq_mhz', 'freq_hz', 'source_folder', 'pi_nu…` |
| `DEFAULT_EDGE_K` | `frozenset({0, 1, 51, 52})` |

## Functions

- **`_k_from_occ(data_dir: Path, design_id: str)`**
- **`load_design_k(data_dir: Path)`** — design_id → decap count K (from shared occupancy).
- **`target_layouts_per_k(k_values: list[int], *, n_ref: float, tau: float, tau_low: float, k_low_max: int, k_anchor: int, n_min: int, n_max: int)`** — Inverse-exponential keep budget per K (capped by available count).
- **`select_designs_to_keep(design_k: dict[str, int], targets: dict[int, int], *, seed: int, drop_edge_k: bool, edge_k: frozenset[int])`** — Return (keep_set, delete_set, per_k_stats).
- **`load_manifest_rows(data_dir: Path)`**
- **`stems_for_designs(rows: list[dict[str, str]], design_ids: set[str])`**
- **`delete_layout_files(data_dir: Path, delete_designs: set[str], delete_stems: set[str], *, execute: bool)`** — Remove layout dirs and per-MHz npy files.
- **`write_manifest(data_dir: Path, rows: list[dict[str, str]], keep_designs: set[str], *, execute: bool)`**
- **`print_plan(design_k: dict[str, int], stats: dict[int, dict[str, int]], n_rows_before: int, n_rows_after: int, file_counts: dict[str, int])`**
- **`main()`**

## Imports

- [[multifreq_layout_store]]
- [[repo_paths]]

## External dependencies

`numpy`, `repo_paths`, `src_vae`
