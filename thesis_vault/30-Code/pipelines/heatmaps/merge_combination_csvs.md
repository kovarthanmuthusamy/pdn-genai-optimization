---
title: merge_combination_csvs
type: code
path: pipelines/heatmaps/merge_combination_csvs.py
group: pipelines/heatmaps
loc: 217
tags: [code, pipelines, runnable]
---

# merge_combination_csvs

> Merge old + new decap layout CSVs for unified PEB / multifreq simulation.

**Source:** `pipelines/heatmaps/merge_combination_csvs.py` · 217 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python pipelines/heatmaps/merge_combination_csvs.py`

## Purpose

```text
Merge old + new decap layout CSVs for unified PEB / multifreq simulation.

Order (required for PI-* alignment):
  1. ``all_combinations.csv``  — 19,499 filtered legacy layouts (peb_row 0..N_old-1)
  2. ``combinations.csv``      — 10,000 new layouts (local row 0..N_new-1)

Merged row ``i`` → ECADStar ``PI-(i+1)``.

Legacy rows use ``decap_index_map.csv`` for ``design_id`` / ``decap_index``.
New rows use ``combinations_pi{pi}_d{local_row}`` with ``pi = local_row + 1``.

Run:
    python pipelines/heatmaps/merge_combination_csvs.py
```

## Constants

| Name | Value |
|------|-------|
| `_REPO_BOOT` | `Path(__file__).resolve().parents[2]` |
| `N_DECAPS` | `52` |
| `OLD_CSV` | `REPO_ROOT / 'data' / 'heatmaps' / 'all_combinations.csv'` |
| `NEW_CSV` | `REPO_ROOT / 'data' / 'heatmaps' / 'combinations.csv'` |
| `DECAP_INDEX_MAP` | `REPO_ROOT / 'data' / 'heatmaps' / 'decap_index_map.csv'` |
| `OUTPUT_CSV` | `REPO_ROOT / 'data' / 'heatmaps' / 'all_combinations_merged.csv'` |
| `OUTPUT_MAP` | `REPO_ROOT / 'data' / 'heatmaps' / 'merged_combinations_index_map.csv'` |
| `OUTPUT_REPORT` | `REPO_ROOT / 'data' / 'heatmaps' / 'merged_combinations_report.json'` |
| `EXECUTE` | `True` |

## Functions

- **`load_rows(path: Path)`**
- **`load_legacy_map(path: Path)`**
- **`merge_combination_csvs(*, old_csv: Path, new_csv: Path, legacy_map_csv: Path, output_csv: Path, output_map: Path, execute: bool)`**
- **`main()`**

## Imports

- [[repo_paths]]

## External dependencies

`repo_paths`
