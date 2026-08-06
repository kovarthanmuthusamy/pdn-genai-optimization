---
title: batch_over_k
type: code
path: scrap/comparison/batch_over_k.py
group: scrap/comparison
loc: 52
tags: [code, scrap]
---

# batch_over_k

> Run a single-K comparison module for K_MIN..K_MAX.

**Source:** `scrap/comparison/batch_over_k.py` · 52 lines

## Functions

- **`run_over_k(module: ModuleType, *, k_min: int, k_max: int, base_dir: str | Path, fail_fast: bool=False, skip_markers: Iterable[str]=('folder not found', 'No real', 'No data_sample'))`** — Patch ``module.K_VALUE`` / ``module.BASE_GENERATED_DIR`` and call ``module.main()`` per K.

## Imported by

- [[compare_generated_vs_real_all_k]]
- [[compare_generated_vs_real_occupancy_all_k]]

## External dependencies

`types`
