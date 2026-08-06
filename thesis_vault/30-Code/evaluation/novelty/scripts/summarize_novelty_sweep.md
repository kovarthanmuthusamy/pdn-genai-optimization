---
title: summarize_novelty_sweep
type: code
path: evaluation/novelty/scripts/summarize_novelty_sweep.py
group: evaluation/novelty/scripts
loc: 716
tags: [code, evaluation, runnable]
---

# summarize_novelty_sweep

> Post-process a novelty sweep folder into a more detailed report.

**Source:** `evaluation/novelty/scripts/summarize_novelty_sweep.py` · 716 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python evaluation/novelty/scripts/summarize_novelty_sweep.py`

## Purpose

```text
Post-process a novelty sweep folder into a more detailed report.

Reads:
- <sweep_root>/novelty_sweep_summary.csv (aggregate medians)
- <sweep_root>/K{K}/novelty_report.csv (per-sample scores)

Writes:
- <sweep_root>/novelty_sweep_summary_detailed.csv
- <sweep_root>/novelty_sweep_summary_detailed.md

This avoids recomputing distances; it summarizes the already-written CSVs.

Additional novelty method (discrete)
- Occupancy-pattern novelty: for each K, measure how often a generated 52-bit
    occupancy pattern was never seen in the training set for that K.
```

## Constants

| Name | Value |
|------|-------|
| `SWEEP_ROOT` | `Path('evaluation/novelty/runs/novelty_sweep_N100')` |
| `DATASET_ROOT` | `Path('datasets/data_norm')` |
| `NO_OCC_NOVELTY` | `False` |
| `GENERATE_PLOTS` | `False` |

## Functions

- **`_q(x: list[float])`**
- **`_fmt(v: float | None)`**
- **`_as_pct(v: float | None)`**
- **`_extract_block(text: str, *, begin: str, end: str)`** — Extract a preserved manual block from an existing markdown file.
- **`_to_float_or_nan(v: Any)`**
- **`_generate_plots(*, rows_by_k: list[dict[str, Any]], out_dir: Path)`** — Generate PNG plots into out_dir.
- **`_occ_key_from_vec(occ_vec: np.ndarray)`**
- **`_entropy_bits(counter: Counter[bytes], total: int)`**
- **`_jsd_bits(p_counter: Counter[bytes], p_total: int, q_counter: Counter[bytes], q_total: int)`**
- **`_load_or_build_k_cache(dataset_root: Path)`**
- **`_build_train_occ_counters(*, dataset_root: Path, ks: set[int])`**
- **`_load_aggregate_summary(path: Path)`**
- **`_load_per_k_report(path: Path)`**
- **`main()`**

## External dependencies

`matplotlib`, `numpy`
