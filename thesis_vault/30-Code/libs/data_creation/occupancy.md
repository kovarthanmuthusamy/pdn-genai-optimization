---
title: occupancy
type: code
path: libs/data_creation/occupancy.py
group: libs/data_creation
loc: 240
tags: [code, libs]
---

# occupancy

> Map 52-d decap vectors to 7×8 physical occupancy grids.

**Source:** `libs/data_creation/occupancy.py` · 240 lines

## Purpose

```text
Map 52-d decap vectors to 7×8 physical occupancy grids.

Purpose:
    Convert binary decap vectors to board-layout grids (4 invalid cells) with C1..C52 labels.

Run:
    Import only — ``from libs.data_creation.occupancy import create_occupancy_grid``.

Agent notes:
    - What: Shared library mapping 52-d binary vectors to physical 7×8 capacitor grid coordinates.
    - Usage: Import ``create_occupancy_grid`` or call ``csv_to_occupancy_samples`` for batch CSV→npy.
    - Key symbols: ``create_occupancy_grid``, ``LABELS_ORDERED``, ``INVALID_OCC_CELLS``
    - Grid layout: origin lower; bottom row C4,C5,…,C1,C2,C3 per board geometry.
```

## Constants

| Name | Value |
|------|-------|
| `INVALID_OCC_CELLS` | `[(0, 3), (0, 4), (6, 3), (6, 4)]` |
| `LABELS_ORDERED` | `['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',…` |
| `OCC_GRID_MAP` | `_create_occupancy_grid_map()` |
| `_LABEL_TO_COORD` | `{label: coord for (coord, label) in OCC_GRID_MAP.items()}` |

## Functions

- **`_create_occupancy_grid_map()`** — Create 7x8 occupancy grid with correct physical position mapping.
- **`create_occupancy_grid(decap_vector)`** — Creates HxW occupancy grid from decap vector - optimized.
- **`visualize_occupancy_grid(occ_grid_file, output_path=None, show=True)`** — Visualize occupancy grid from saved numpy file - raw binary display.
- **`csv_row_to_occupancy(csv_row, expected_size=56)`** — Directly convert a CSV row to occupancy grid data without complex mapping.
- **`csv_to_occupancy_samples(csv_file, output_dir, start_idx=0)`** — Batch convert CSV rows directly to occupancy samples.

## Imported by

- [[experiments.exp043.codes.vae_multi_input_simple]]
- [[experiments.exp045.codes.vae_multi_input_simple]]
- [[experiments.exp046.codes.vae_multi_input_simple]]
- [[experiments.exp047.codes.vae_multi_input_simple]]
- [[experiments.exp048.codes.vae_multi_input_simple]]
- [[experiments.exp049.codes.vae_multi_input_simple]]
- [[experiments.exp050.codes.vae_multi_input_simple]]
- [[experiments.exp051_new_datas_appended.codes.vae_multi_input_simple]]
- [[experiments.exp052_unbounded_pearson.codes.vae_multi_input_simple]]
- [[experiments.exp053_peak_log1p_losses.codes.vae_multi_input_simple]]
- [[experiments.exp054_K_30.codes.vae_multi_input_simple]]
- [[experiments.exp055_hard_occ.codes.vae_multi_input_simple]]
- [[experiments.exp056_graph_vae.codes.graph_occ]]
- [[experiments.exp056_graph_vae.codes.vae_multi_input_simple]]
- [[experiments.exp057_structured_graph.codes.graph_occ]]
- [[experiments.exp057_structured_graph.codes.vae_multi_input_simple]]
- [[experiments.exp058_asymmetric_kl.codes.graph_occ]]
- [[experiments.exp058_asymmetric_kl.codes.vae_multi_input_simple]]
- [[experiments.exp059_capacity_freq.codes.graph_occ]]
- [[experiments.exp059_capacity_freq.codes.vae_multi_input_simple]]
- [[experiments.exp060_multitype_occ.codes.graph_occ]]
- [[experiments.exp060_multitype_occ.codes.vae_multi_input_simple]]

## External dependencies

`matplotlib`, `numpy`
