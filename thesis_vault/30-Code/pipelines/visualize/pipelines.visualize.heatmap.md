---
title: heatmap
type: code
path: pipelines/visualize/heatmap.py
group: pipelines/visualize
loc: 207
tags: [code, pipelines, runnable]
---

# heatmap

> EM solver MAP file to heatmap visualizer.

**Source:** `pipelines/visualize/heatmap.py` · 207 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python pipelines/visualize/heatmap.py`

## Purpose

```text
EM solver MAP file to heatmap visualizer.

Purpose:
    Parse a Z_*.map point cloud, interpolate onto a 64×64 grid with H-shaped board mask,
    and render impedance distribution heatmaps.

Run:
    python pipelines/visualize/heatmap.py

Agent notes:
    - What: Converts ECAD MAP text to numpy heatmap grids and PNG plots.
    - Usage: Set ``MAP_FILE`` in CONFIG → run demo, or import ``plot_heatmap_array`` from other scripts.
    - Config keys:
        - ``MAP_FILE`` — path to ``Z_*.map`` frequency point cloud
    - Key symbols: ``plot_impedance_heatmap_clean``, ``plot_heatmap_array``
```

## Constants

| Name | Value |
|------|-------|
| `MAP_FILE` | `repo_path('scripts', 'Z_0063.000MHz.map')` |

## Functions

- **`plot_impedance_heatmap_clean(file_path)`**
- **`plot_heatmap_array(heatmap, output_path, title='Generated Heatmap', mask=None, threshold=0.5, vmin=None, vmax=None)`**

## Imports

- [[repo_paths]]

## External dependencies

`matplotlib`, `numpy`, `repo_paths`, `scipy`
