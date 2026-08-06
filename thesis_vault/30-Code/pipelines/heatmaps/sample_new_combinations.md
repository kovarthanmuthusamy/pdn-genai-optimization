---
title: sample_new_combinations
type: code
path: pipelines/heatmaps/sample_new_combinations.py
group: pipelines/heatmaps
loc: 427
tags: [code, pipelines, runnable]
---

# sample_new_combinations

> Sample new decap layout combinations for PI simulation (inverse-K, exclude existing).

**Source:** `pipelines/heatmaps/sample_new_combinations.py` · 427 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python pipelines/heatmaps/sample_new_combinations.py`

## Purpose

```text
Sample new decap layout combinations for PI simulation (inverse-K, exclude existing).

Reads the existing ``all_combinations.csv``, then writes ``combinations.csv`` with
``TOTAL_N`` new 52-d binary rows **not** present in the old file.

- **K=2:** every remaining unique pair layout (all C(52,2) not already in the old CSV).
- **K=3..50:** inverse-exponential counts for the rest of ``TOTAL_N``.

Run:
    python pipelines/heatmaps/sample_new_combinations.py

Agent notes:
    - Old layouts: ``EXISTING_CSV`` (default ``data/heatmaps/all_combinations.csv``)
    - Output: ``OUTPUT_CSV`` (default ``data/heatmaps/combinations.csv``)
    - Optional ``CANDIDATE_POOL_CSV``: larger CSV to draw from instead of random gen
```

## Constants

| Name | Value |
|------|-------|
| `_REPO_BOOT` | `Path(__file__).resolve().parents[2]` |
| `EXISTING_CSV` | `_ROOT / 'data' / 'heatmaps' / 'all_combinations.csv'` |
| `OUTPUT_CSV` | `_ROOT / 'data' / 'heatmaps' / 'combinations.csv'` |
| `REPORT_JSON` | `_ROOT / 'data' / 'heatmaps' / 'combinations_sample_report.json'` |
| `TOTAL_N` | `10000` |
| `SEED` | `42` |
| `N_DECAPS` | `52` |
| `N_REF` | `1000.0` |
| `TAU` | `12.0` |
| `TAU_LOW` | `35.0` |
| `K_LOW_MAX` | `20` |
| `K_ANCHOR` | `2` |
| `N_MIN` | `150` |
| `N_MAX` | `1000` |
| `K_MIN` | `2` |
| `K_INVERSE_MIN` | `3` |
| `K_MAX` | `50` |
| `EDGE_K` | `frozenset({0, 1, 51, 52})` |
| `INCLUDE_ALL_K2` | `True` |
| `MAX_GEN_ATTEMPTS_PER_SLOT` | `500000` |

## Functions

- **`row_tuple(values: list[int] | np.ndarray)`**
- **`row_k(values: list[int] | np.ndarray)`**
- **`load_csv_rows(path: Path)`**
- **`comb_capacity(k: int)`** — Number of distinct K-hot layouts on N_DECAPS sites.
- **`existing_counts_by_k(existing: set[tuple[int, ...]])`**
- **`cap_targets_by_availability(targets: dict[int, int], existing_by_k: Counter[int])`** — Cap per-K targets by C(52,K) minus already-used layouts; redistribute deficit.
- **`inverse_k_targets(total_n: int, *, k_min: int=K_INVERSE_MIN, k_max: int=K_MAX, k_anchor: int | None=None)`** — Per-K counts scaled to ``total_n`` (inverse-exponential over k_min..k_max).
- **`all_remaining_k2_layouts(exclude: set[tuple[int, ...]])`** — Enumerate every K=2 layout not in ``exclude`` (C(52,2) pairs).
- **`build_targets(existing_by_k: Counter[int], total_n: int)`** — K=2 = all available; K=3..50 = inverse-K on the remainder of ``total_n``.
- **`random_k_hot(k: int, rng: random.Random)`**
- **`sample_from_pool(pool_by_k: dict[int, list[tuple[int, ...]]], targets: dict[int, int], *, exclude: set[tuple[int, ...]], rng: random.Random)`**
- **`generate_random(targets: dict[int, int], *, exclude: set[tuple[int, ...]], rng: random.Random)`**
- **`write_csv(path: Path, rows: list[tuple[int, ...]])`**
- **`main()`**

## Imports

- [[repo_paths]]

## External dependencies

`numpy`, `repo_paths`
