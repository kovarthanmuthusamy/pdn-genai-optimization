---
title: vae_novelty_report
type: code
path: evaluation/novelty/scripts/vae_novelty_report.py
group: evaluation/novelty/scripts
loc: 728
tags: [code, evaluation, runnable]
---

# vae_novelty_report

> Nearest-neighbor novelty / memorization check for this repo's multi-modal VAE.

**Source:** `evaluation/novelty/scripts/vae_novelty_report.py` · 728 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python evaluation/novelty/scripts/vae_novelty_report.py`

## Purpose

```text
Nearest-neighbor novelty / memorization check for this repo's multi-modal VAE.

Goal
- For each generated sample, find the closest training sample (kNN with k=1)
  and report distances for:
    - pooled heatmap z-score (MSE)
    - impedance log-zscore (MSE)
    - occupancy vector (Hamming distance)
  plus a simple combined score.

Interpretation (rule of thumb)
- If occupancy matches *exactly* (Hamming=0) and the continuous distances are
  extremely small, you're likely memorizing.
- Compare generated→train NN distances to train→train NN baseline distances.
  If generated distances are consistently much smaller than baseline, that is
  suspicious.

This is not a proof of "novel" physics; it is a practical "are we copying the
training set?" sanity check.

Additional novelty method (discrete)
- Occupancy-pattern novelty: for a fixed K, treat the 52-bit occupancy vector
    as a discrete pattern and measure how often generated patterns were never
    seen in the training set for that K.

Example
  python scripts/vae_novelty_report.py     --K 5
```

## Constants

| Name | Value |
|------|-------|
| `GEN_DIR` | `Path('evaluation/novelty/runs/K5_noveltyN50')` |
| `DATASET_ROOT` | `Path('datasets/data_norm')` |
| `HM_POOL` | `16` |
| `BASELINE_N` | `200` |
| `SEED` | `0` |

## Classes

- **`Quantiles(TypedDict)`**
- **`OccupancyPatternStats(TypedDict)`**
- **`DatasetArrays`**
- **`ScoreSummary`**

## Functions

- **`_infer_k_from_path(path: Path)`**
- **`_iter_sample_dirs(gen_dir: Path)`**
- **`_load_json(path: Path)`**
- **`_k_cache_path(dataset_root: Path)`**
- **`_load_or_build_k_cache(dataset_root: Path, *, force_rebuild: bool=False)`** — Return (stems, k_values) for the dataset, cached to disk.
- **`_pool_heatmap_zscore(hm: np.ndarray, pool: int)`** — Mean-pool a (64,64) or (1,64,64) heatmap into (pool,pool), then flatten.
- **`_load_occ_bin(path: Path)`**
- **`_load_impedance_channel0_z(dataset_imp_path: Path)`** — Load dataset impedance and return channel 0 z-score vector (231,).
- **`_occ_keys(occ_bin: np.ndarray)`**
- **`_entropy_bits(counter: Counter[bytes], total: int)`**
- **`_jsd_bits(p_counter: Counter[bytes], p_total: int, q_counter: Counter[bytes], q_total: int)`**
- **`_occupancy_pattern_stats(train_occ: np.ndarray, gen_occ: np.ndarray)`**
- **`_imp_log_to_z(imp_log: np.ndarray, *, log_mean: float, log_std: float)`**
- **`_mse_matrix(gen: np.ndarray, train: np.ndarray)`** — Return MSE matrix (M,N) for L2 in feature space, using dot-product identity.
- **`_hamming_matrix(gen_occ: np.ndarray, train_occ: np.ndarray)`** — Compute (M,N) Hamming distances for 52-bit occupancy. Uses a loop over M.
- **`load_dataset(*, dataset_root: Path, hm_pool: int, max_train: Optional[int], k_filter: Optional[int], seed: int)`**
- **`_quantiles(x: np.ndarray)`**
- **`_percentiles_against_baseline(values: np.ndarray, baseline: np.ndarray)`** — Return percentile ranks in [0,1] for values relative to baseline.
- **`_fmt_q(d: Quantiles)`**
- **`score_generated_against_dataset(*, gen_dir: Path, dataset_root: Path, k_filter: int | None, hm_pool: int, baseline_n: int, seed: int, max_gen: int | None=None, max_train: int | None=None, out_csv: Path | None=None)`** — Compute novelty scores and optionally write a CSV (same format as ``vae_novelty_report.py`` main).
- **`load_generated(*, gen_dir: Path, hm_pool: int, max_gen: Optional[int], imp_log_mean: float, imp_log_std: float)`**
- **`_describe_dist(x: np.ndarray)`**
- **`main()`**

## Imported by

- [[run_vae_novelty_test]]

## External dependencies

`numpy`
