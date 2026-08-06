---
title: candidates
type: code
path: active_learning_pi/al/candidates.py
group: active_learning_pi/al
loc: 244
tags: [code, active_learning_pi]
---

# candidates

> Random decap-layout × PI-frequency candidate pool generation.

**Source:** `active_learning_pi/al/candidates.py` · 244 lines

## Purpose

```text
Random decap-layout × PI-frequency candidate pool generation.

Run:
    python active_learning_pi/al/candidates.py
```

## Classes

- **`Candidate`**

## Functions

- **`random_occupancy(k: int, rng: np.random.Generator, n_decaps: int=52)`**
- **`build_mhz_schedule(strata: dict[float, int], num_candidates: int, rng: np.random.Generator)`** — Expand MHz→count strata to a shuffled per-candidate MHz list (length ``num_candidates``).
- **`_scale_strata_to_target(strata: dict[float, int], target_n: int)`** — Scale MHz→count strata to sum to ``target_n`` while preserving proportions.
- **`_quantize_mhz(x: float, step: float)`**
- **`_sample_explore_mhz(*, rng: np.random.Generator, n: int, explore_mhz_grid: list[float] | None=None, explore_mhz_min: float | None=None, explore_mhz_max: float | None=None, explore_mhz_quantize: float=0.0, explore_mhz_band_edges: list[float] | None=None, avoid_mhz: set[float] | None=None)`**
- **`_mhz_key(mhz: float)`**
- **`generate_candidates(*, num_candidates: int, mhz_grid: list[float], fixed_k: int | None=None, k_values: list[int] | None=None, seed: int=42, mhz_priority: list[float] | None=None, mhz_strata: dict[float, int] | None=None, explore_mhz_grid: list[float] | None=None, explore_n: int=0, explore_mhz_min: float | None=None, explore_mhz_max: float | None=None, explore_mhz_quantize: float=0.0, explore_mhz_band_edges: list[float] | None=None)`** — Sample (occupancy, MHz) pairs: random K-hot decap layouts × frequencies.

## Imported by

- [[evaluate_cycle]]
- [[inference_pool]]
- [[per_k_acquire]]
- [[pipeline]]
- [[validate_acquisition_ab]]

## External dependencies

`numpy`
