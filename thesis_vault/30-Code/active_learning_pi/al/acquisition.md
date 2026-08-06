---
title: acquisition
type: code
path: active_learning_pi/al/acquisition.py
group: active_learning_pi/al
loc: 268
tags: [code, active_learning_pi]
---

# acquisition

> Acquisition functions — select worst/uncertain candidates for ECADSTAR simulation.

**Source:** `active_learning_pi/al/acquisition.py` · 268 lines

## Purpose

```text
Acquisition functions — select worst/uncertain candidates for ECADSTAR simulation.

Run:
    python active_learning_pi/al/acquisition.py
```

## Functions

- **`_hamming(a: list[int], b: list[int])`**
- **`add_badness_scores(scored: list[dict[str, Any]])`** — badness = how much we need ECADStar labels (higher = worse / more uncertain).
- **`select_worst_for_simulation(scored: list[dict[str, Any]], *, simulate_batch_size: int, min_uncertainty: float | None=None, min_uncertainty_percentile: float=0.0, stratify_by_k: bool=False, mhz_quotas: dict[float, int] | None=None)`** — Pick only the worst candidates (highest uncertainty) for a single ECADStar batch.
- **`_select_worst_stratified_by_k(scored: list[dict[str, Any]], *, n: int, min_uncertainty: float | None, min_uncertainty_percentile: float)`**
- **`_mhz_match(mhz: float)`**
- **`_select_worst_stratified_by_mhz(scored: list[dict[str, Any]], *, mhz_quotas: dict[float, int], n: int, min_uncertainty: float | None, min_uncertainty_percentile: float)`**
- **`select_batch(scored: list[dict[str, Any]], *, batch_size: int, top_n_fraction: float=0.6, random_fraction: float=0.2, diversity_fraction: float=0.2, seed: int=0)`** — Legacy mixed acquisition (optional; not used for ECADStar batch by default).

## Imported by

- [[per_k_acquire]]
- [[pipeline]]

## External dependencies

`numpy`
