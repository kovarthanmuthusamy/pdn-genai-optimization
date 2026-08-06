---
title: change_frequency
type: code
path: pipelines/heatmaps/change_frequency.py
group: pipelines/heatmaps
loc: 52
tags: [code, pipelines, runnable]
---

# change_frequency

> Change PI-Distribution frequency in a single .peb file.

**Source:** `pipelines/heatmaps/change_frequency.py` · 52 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python pipelines/heatmaps/change_frequency.py`

## Purpose

```text
Change PI-Distribution frequency in a single .peb file.

Purpose:
    Replace ``EditPIDistribution Frequency="..."`` in a source PEB and write one output file.

Run:
    python pipelines/heatmaps/change_frequency.py

Agent notes:
    - What: Single-frequency PEB rewrite (one MHz). For all anchors use ``regenerate_mhz_pebs.py``.
    - Usage: Set ``INPUT_PEB``, ``SET_FREQ_MHZ``, ``OUTPUT_PEB`` → run.
    - Config keys:
        - ``INPUT_PEB`` — source PEB path
        - ``SET_FREQ_MHZ`` — target inspection frequency (MHz)
        - ``OUTPUT_PEB`` — destination path
```

## Constants

| Name | Value |
|------|-------|
| `_REPO_BOOT` | `Path(__file__).resolve().parents[2]` |
| `SCRIPT_DIR` | `repo_path('data', 'heatmaps')` |
| `INPUT_PEB` | `SCRIPT_DIR / 'combined_merged_63MHz.peb'` |
| `SET_FREQ_MHZ` | `0` |
| `OUTPUT_PEB` | `SCRIPT_DIR / f'peb_with_29k/combined_all_{SET_FREQ_MHZ}MHz.peb'` |

## Functions

- **`main()`**

## Imports

- [[frequency]]
- [[repo_paths]]

## External dependencies

`libs`, `repo_paths`
