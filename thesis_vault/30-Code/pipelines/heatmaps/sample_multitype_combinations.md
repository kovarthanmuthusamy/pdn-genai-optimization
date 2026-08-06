---
title: sample_multitype_combinations
type: code
path: pipelines/heatmaps/sample_multitype_combinations.py
group: pipelines/heatmaps
loc: 577
tags: [code, pipelines, runnable, uncommitted]
---

# sample_multitype_combinations

> Sample multi-type decap layout CSVs (type codes, K <= 5).

**Source:** `pipelines/heatmaps/sample_multitype_combinations.py` · 577 lines
**Git:** uncommitted — not yet tracked
**Runnable:** CONFIG-only script — edit constants at top, then `python pipelines/heatmaps/sample_multitype_combinations.py`

## Purpose

```text
Sample multi-type decap layout CSVs (type codes, K <= 5).

Primary output (COMBINED mode, default): a **single** CSV mixing three families
of layouts in one file (52 columns, integer type codes per slot):

1. **Type-1 only** — occupied slots are all ``1``.
2. **Type-2 only** — occupied slots are all ``2``.
3. **Mixed types** — each row uses both type ``1`` and type ``2``.

The combined file holds ``TOTAL_COMBINED`` (default 15000) unique random rows,
split evenly across the three families and across K = 2..5, then shuffled.

Schema matches exp060 occupancy (no empty channel in the model tensor;
CSV stores integer codes ``0/1/2``):

    0 = empty, 1 = type-1, 2 = type-2

Legacy per-family CSVs (type-2 only + mixed) are still written when
``WRITE_LEGACY_SPLIT`` is True.

Run:
    python pipelines/heatmaps/sample_multitype_combinations.py
```

## Constants

| Name | Value |
|------|-------|
| `_REPO_BOOT` | `Path(__file__).resolve().parents[2]` |
| `N_DECAPS` | `52` |
| `TYPE1` | `1` |
| `TYPE2` | `2` |
| `K_MIN_TYPE2` | `2` |
| `K_MAX` | `5` |
| `K_MIN_MIXED` | `2` |
| `SEED` | `42` |
| `USE_MATCH_TYPE1_COUNTS` | `True` |
| `N_PER_K_TYPE2` | `500` |
| `N_PER_K_MIXED` | `500` |
| `OUTPUT_TYPE2_CSV` | `_ROOT / 'data' / 'heatmaps' / 'combinations_type2_only_K2_5.csv'` |
| `OUTPUT_MIXED_CSV` | `_ROOT / 'data' / 'heatmaps' / 'combinations_mixed_types_K2_5.csv'` |
| `REPORT_JSON` | `_ROOT / 'data' / 'heatmaps' / 'combinations_multitype_sample_report.json'` |
| `WRITE_COMBINED` | `True` |
| `WRITE_LEGACY_SPLIT` | `False` |
| `TOTAL_COMBINED` | `15000` |
| `INCLUDE_K1_SINGLE` | `True` |
| `K_MAX_COMBINED` | `K_MAX` |
| `SHUFFLE_COMBINED` | `True` |
| `OUTPUT_COMBINED_CSV` | `_ROOT / 'data' / 'heatmaps' / 'combinations_multitype_combined_15000.csv'` |
| `MAX_GEN_ATTEMPTS_PER_K` | `2000000` |
| `_SINGLE_TYPE_CODE` | `{random_type1_only: TYPE1, random_type2_only: TYPE2}` |

## Functions

- **`row_k(row: tuple[int, ...] | list[int])`**
- **`is_type1_only(row: tuple[int, ...])`**
- **`is_type2_only(row: tuple[int, ...])`**
- **`is_mixed(row: tuple[int, ...])`**
- **`write_csv(path: Path, rows: list[tuple[int, ...]])`**
- **`targets_per_k(*, k_min: int, k_max: int, n_per_k: int, total: int | None)`**
- **`decreasing_per_k(k_start: int, k_max: int, total: int)`** — Allocate ``total`` over K=k_start..k_max with **decreasing** counts.
- **`type2_capacity(k: int)`** — Distinct type-2-only layouts at budget K (= C(52,K)).
- **`mixed_capacity_lower_bound(k: int)`** — Loose lower bound: choose slots, then 2^K - 2 non-monochrome labelings.
- **`cap_targets(targets: dict[int, int], capacity_fn)`** — Cap each K by capacity; redistribute shortfall to K with spare room.
- **`_random_single_type(k: int, rng: random.Random, code: int)`**
- **`random_type1_only(k: int, rng: random.Random)`**
- **`random_type2_only(k: int, rng: random.Random)`**
- **`_all_single_type_k(k: int, code: int)`** — Enumerate every single-type layout (given code) with exactly K slots.
- **`all_type2_only_k(k: int)`** — Enumerate every type-2-only layout with exactly K occupied slots.
- **`all_mixed_k(k: int)`** — Enumerate every mixed layout (both types present) with K occupied slots.
- **`random_mixed(k: int, rng: random.Random)`** — K occupied slots with both types present (rejection if not mixed).
- **`generate_unique(targets: dict[int, int], *, factory, validate, rng: random.Random, label: str)`**
- **`_summarize(rows: list[tuple[int, ...]])`**
- **`build_combined(rng: random.Random)`** — Generate one merged list of TOTAL_COMBINED rows with the target K-shape.
- **`main()`**

## Imports

- [[repo_paths]]

## External dependencies

`repo_paths`
