---
title: generate_peb
type: code
path: scrap/generation/generate_peb.py
group: scrap/generation
loc: 124
tags: [code, scrap]
---

# generate_peb

> Generate ECADStar Batch PEB.

**Source:** `scrap/generation/generate_peb.py` · 124 lines

## Purpose

```text
Generate ECADStar Batch PEB.

Run: import generate_peb(...) or run as script with INPUT_PATH/OUTPUT_PATH constants.
```

## Constants

| Name | Value |
|------|-------|
| `INPUT_PATH` | `'occupancy.npy'` |
| `OUTPUT_PATH` | `'new.peb'` |
| `POWERBUS` | `'Power_GND'` |
| `FREQ` | `'63e6'` |
| `COMPONENTS` | `'IC1_Port1,IC2_Port2'` |
| `N_COMPONENTS` | `52` |

## Functions

- **`build_edit_lines(occ_row: np.ndarray, indent: str='      ')`** — Return XML Edit lines for all 52 components based on a 0/1 occupancy row.
- **`build_distribution_group(occ_row: np.ndarray, freq: str)`**
- **`build_spectrum_group(occ_row: np.ndarray, components: str)`**
- **`generate_peb(occupancy: np.ndarray, output_path: str, powerbus: str='Power_GND', freq: str='63e6', components: str='IC1_Port1,IC2_Port2', per_sample_freqs: list[str] | None=None, include_distribution: bool=True, include_spectrum: bool=True)`** — occupancy        : np.ndarray of shape (N, 52) with 0/1 values

## Imported by

- [[generate_samples_and_peb]]
- [[peb]]
- [[peb_batch]]
- [[run_all_k]]
- [[run_multifreq_heatmap_sweep]]
- [[scrap_pipeline]]

## External dependencies

`numpy`
