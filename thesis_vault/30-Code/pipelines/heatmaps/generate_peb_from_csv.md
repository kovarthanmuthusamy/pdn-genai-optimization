---
title: generate_peb_from_csv
type: code
path: pipelines/heatmaps/generate_peb_from_csv.py
group: pipelines/heatmaps
loc: 112
tags: [code, pipelines, runnable]
---

# generate_peb_from_csv

> Generate ECADStar .peb file(s) from a decap combinations CSV (52-column 0/1 rows).

**Source:** `pipelines/heatmaps/generate_peb_from_csv.py` · 112 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python pipelines/heatmaps/generate_peb_from_csv.py`

## Purpose

```text
Generate ECADStar .peb file(s) from a decap combinations CSV (52-column 0/1 rows).

Each CSV row → one PI job in PEB order (row 0 = PI-1, row N-1 = PI-N).

Run:
    python pipelines/heatmaps/generate_peb_from_csv.py

Typical use after merge:
    INPUT_CSV = data/heatmaps/all_combinations_merged.csv
    → combined_merged_63MHz.peb  (PI-Distribution @ 63 MHz)
    → combined_merged_impedance.peb  (CreatePISpectrum only)

For per-anchor MHz files from an existing master PEB, use ``regenerate_mhz_pebs.py``
or ``change_frequency.py`` instead.
```

## Constants

| Name | Value |
|------|-------|
| `_REPO_BOOT` | `Path(__file__).resolve().parents[2]` |
| `INPUT_CSV` | `repo_path('data', 'heatmaps', 'all_combinations_merged.csv')` |
| `OUTPUT_DIR` | `repo_path('data', 'heatmaps')` |
| `DIST_MHZ` | `63.0` |
| `WRITE_DISTRIBUTION` | `True` |
| `WRITE_IMPEDANCE` | `True` |
| `POWERBUS` | `'Power_GND'` |
| `IMPEDANCE_COMPONENTS` | `'IC1_Port1,IC2_Port2'` |
| `DIST_PEB_NAME` | `'combined_merged_{mhz}MHz.peb'` |
| `IMPEDANCE_PEB_NAME` | `'combined_merged_impedance.peb'` |
| `START_ROW` | `0` |

## Functions

- **`_slice_occupancy(occ, start: int, max_rows: int | None)`**
- **`main()`**

## Imports

- [[combinations]]
- [[peb]]
- [[repo_paths]]

## External dependencies

`repo_paths`
