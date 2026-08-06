---
title: run_vae_novelty_sweep
type: code
path: evaluation/novelty/scripts/run_vae_novelty_sweep.py
group: evaluation/novelty/scripts
loc: 220
tags: [code, evaluation, runnable]
---

# run_vae_novelty_sweep

> Generate + score novelty for many K values.

**Source:** `evaluation/novelty/scripts/run_vae_novelty_sweep.py` · 220 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python evaluation/novelty/scripts/run_vae_novelty_sweep.py`

## Purpose

```text
Generate + score novelty for many K values.

This is the "include most K and increase N" runner.

Outputs
- Per-K folders under:
    <out-root>/K{K}/data_sample_i/{heatmap_zscore.npy, occupancy_map.npy, impedance_profile.npy}
  plus per-K `novelty_report.csv`.
- One aggregate CSV:
    <out-root>/novelty_sweep_summary.csv

Example

Tip
- Set MAX_TRAIN in CONFIG to cap dataset size per K for speed.
```

## Constants

| Name | Value |
|------|-------|
| `CHECKPOINT` | `'experiments/exp027_sigma_reg_tuning/checkpoints/checkpoint_epoch_400.pt'` |
| `LATENT_DIM` | `32` |
| `NUM_SAMPLES_PER_K` | `100` |
| `K_START` | `1` |
| `K_END` | `52` |
| `K_STEP` | `1` |
| `SHARED_TEMP` | `1.5` |
| `DATASET_ROOT` | `Path('datasets/data_norm')` |
| `HM_POOL` | `16` |
| `BASELINE_N` | `200` |
| `SEED` | `0` |
| `SCORE_ONLY` | `False` |
| `FORCE_CPU` | `False` |

## Functions

- **`_project_root()`**
- **`_load_engine(*, checkpoint: str, latent_dim: int, force_cpu: bool)`**
- **`_generate_one_k(*, engine, out_dir: Path, k_value: int, n: int, shared_temp: float)`**
- **`main()`**

## External dependencies

`experiments`, `importlib`, `numpy`, `torch`
