---
title: combinations
type: code
path: pipelines/dataset_sim/combinations.py
group: pipelines/dataset_sim
loc: 33
tags: [code, pipelines]
---

# combinations

> Load decap combination rows from CSV.

**Source:** `pipelines/dataset_sim/combinations.py` · 33 lines

## Constants

| Name | Value |
|------|-------|
| `N_DECAPS` | `52` |

## Functions

- **`load_combinations_csv(path: Path)`** — Return (N, 52) int8 occupancy rows.
- **`layout_dir_name(index: int)`**

## Imported by

- [[append_legacy_19k_multifreq]]
- [[append_merged_combinations_multifreq]]
- [[append_restore_49k_legacy_multifreq]]
- [[generate_peb_from_csv]]
- [[peb]]
- [[run_combinations_sim_pipeline]]

## External dependencies

`numpy`
