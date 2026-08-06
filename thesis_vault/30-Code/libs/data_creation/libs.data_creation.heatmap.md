---
title: heatmap
type: code
path: libs/data_creation/heatmap.py
group: libs/data_creation
loc: 120
tags: [code, libs]
---

# heatmap

> Parse ECADStar .map files and interpolate PI-Distribution heatmaps (64×64).

**Source:** `libs/data_creation/heatmap.py` · 120 lines

## Purpose

```text
Parse ECADStar .map files and interpolate PI-Distribution heatmaps (64×64).

Run:
    Import only — called by ``Data_processing*.py``, ``ingest_labels``, comparison scripts.
```

## Constants

| Name | Value |
|------|-------|
| `HEATMAP_GRID_SIZE` | `64` |

## Functions

- **`load_mask_board(path)`** — Load mask board from npy file.
- **`create_Heatmaps(file_path, grid_size=HEATMAP_GRID_SIZE, frame_path=None, mask_board=None, verbose=False)`** — Creates stacked heatmap with impedance and mask channels - optimized.
- **`visualize_heatmap(heatmap_file, output_path=None, show=True)`** — Visualize heatmap channels from saved numpy file - raw data display.

## Imported by

- [[_analyze_hm_pattern]]
- [[append_legacy_19k_multifreq]]
- [[append_merged_combinations_multifreq]]
- [[append_restore_49k_legacy_multifreq]]
- [[ingest_labels]]
- [[processing_eval]]
- [[processing_multifreq]]
- [[visualize_sample]]

## External dependencies

`matplotlib`, `numpy`, `scipy`
