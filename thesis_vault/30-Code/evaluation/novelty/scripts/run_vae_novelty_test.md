---
title: run_vae_novelty_test
type: code
path: evaluation/novelty/scripts/run_vae_novelty_test.py
group: evaluation/novelty/scripts
loc: 217
tags: [code, evaluation, runnable]
---

# run_vae_novelty_test

> End-to-end novelty test: generate N samples + score vs dataset.

**Source:** `evaluation/novelty/scripts/run_vae_novelty_test.py` · 217 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python evaluation/novelty/scripts/run_vae_novelty_test.py`

## Purpose

```text
End-to-end novelty test: generate N samples + score vs dataset.

This script:
1) Uses the exp027 inference code to generate a small batch of samples for a given K.
2) Saves them in the same on-disk layout as `scrap/generate_samples_and_peb.py`.
3) Runs `scripts/vae_novelty_report.py` to compute nearest-neighbor scores and write a CSV.
4) Writes a short markdown summary next to the CSV.

Example
  python scripts/run_vae_novelty_test.py

Notes
- This does NOT require ECADStar / .peb.
- For scoring, we strongly recommend filtering the dataset to the same K.
```

## Constants

| Name | Value |
|------|-------|
| `CHECKPOINT` | `'experiments/exp027_sigma_reg_tuning/checkpoints/checkpoint_epoch_400.pt'` |
| `LATENT_DIM` | `32` |
| `K_VALUE` | `5` |
| `NUM_SAMPLES` | `50` |
| `SHARED_TEMP` | `1.5` |
| `DATASET_ROOT` | `Path('datasets/data_norm')` |
| `HM_POOL` | `16` |
| `BASELINE_N` | `200` |
| `FORCE_CPU` | `False` |

## Functions

- **`_project_root()`**
- **`_python_exe()`**
- **`_save_generated_samples(*, out_dir: Path, checkpoint_path: str, latent_dim: int, k_value: int, num_samples: int, shared_temp: float, force_cpu: bool)`**
- **`_run_report(*, gen_dir: Path, dataset_root: Path, k_value: int, hm_pool: int, baseline_n: int, max_train: int | None)`**
- **`_write_summary(*, out_dir: Path, params: dict, report_stdout: str)`**
- **`main()`**

## Imports

- [[vae_novelty_report]]

## External dependencies

`experiments`, `numpy`, `torch`
