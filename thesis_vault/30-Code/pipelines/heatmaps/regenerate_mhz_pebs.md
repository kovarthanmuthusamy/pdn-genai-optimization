---
title: regenerate_mhz_pebs
type: code
path: pipelines/heatmaps/regenerate_mhz_pebs.py
group: pipelines/heatmaps
loc: 52
tags: [code, pipelines, runnable]
---

# regenerate_mhz_pebs

> Batch-regenerate anchor-frequency PEB files from combined_all.peb.

**Source:** `pipelines/heatmaps/regenerate_mhz_pebs.py` · 52 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python pipelines/heatmaps/regenerate_mhz_pebs.py`

## Purpose

```text
Batch-regenerate anchor-frequency PEB files from combined_all.peb.

Purpose:
    Write ``combined_all_{MHz}MHz.peb`` for each MHz in ``ANCHORS_MHZ`` by rewriting PI frequency tags.

Run:
    python pipelines/heatmaps/regenerate_mhz_pebs.py

Agent notes:
    - What: Splits one master PEB into per-anchor-frequency PEB files for ECADStar batch runs.
    - Usage: Point ``INPUT_PEB`` at ``combined_all.peb`` → set ``ANCHORS_MHZ`` → run.
    - Config keys:
        - ``INPUT_PEB`` — source combined PEB under ``data/heatmaps/``
        - ``ANCHORS_MHZ`` — list of MHz values to emit
    - Key symbol: ``write_peb_at_mhz`` (from ``libs.peb.frequency``)
```

## Constants

| Name | Value |
|------|-------|
| `_REPO_BOOT` | `Path(__file__).resolve().parents[2]` |
| `SCRIPT_DIR` | `repo_path('data', 'heatmaps')` |
| `INPUT_PEB` | `SCRIPT_DIR / 'combined_all.peb'` |
| `ANCHORS_MHZ` | `[10, 80, 130, 150, 200, 230, 250, 270, 300, 330, 400, 450, 500, 550, 600]` |

## Functions

- **`main()`**

## Imports

- [[frequency]]
- [[repo_paths]]

## External dependencies

`libs`, `repo_paths`
