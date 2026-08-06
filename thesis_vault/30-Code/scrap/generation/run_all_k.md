---
title: run_all_k
type: code
path: scrap/generation/run_all_k.py
group: scrap/generation
loc: 237
tags: [code, scrap, runnable]
---

# run_all_k

> Generate VAE Samples for K Sweep.

**Source:** `scrap/generation/run_all_k.py` · 237 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python scrap/generation/run_all_k.py`

## Purpose

```text
Generate VAE Samples for K Sweep.

Run: python scrap/generation/run_all_k.py
```

## Constants

| Name | Value |
|------|-------|
| `EXPERIMENT_DIR` | `'experiments/exp041'` |
| `CHECKPOINT_PATH` | `f'{EXPERIMENT_DIR}/checkpoints/last_model.pt'` |
| `OUTPUT_ROOT` | `f'{EXPERIMENT_DIR}/generated_samples'` |
| `PEB_OUT_FILE` | `f'{OUTPUT_ROOT}/K1_to_K52.peb'` |
| `K_MIN` | `10` |
| `K_MAX` | `11` |
| `NUM_SAMPLES` | `1` |
| `SHARED_TEMP` | `1.5` |
| `LATENT_STATS_PATH` | `''` |
| `MODEL_LATENT_DIM` | `32` |
| `POWERBUS` | `'Power_GND'` |
| `FREQ` | `'63e6'` |
| `COMPONENTS` | `'IC1_Port1'` |
| `FORCE_CPU` | `False` |

## Functions

- **`_mhz_to_norm(mhz: int)`** — Convert MHz integer to the log10-normalised [0,1] value used by the model.
- **`_generate_save(engine: Any, *, num_samples: int, out_dir: Path, K: int, shared_temp: float, pi_freq: float | None=None)`** — Generate `num_samples` samples with exactly K decaps and save to *out_dir*.
- **`main()`**

## Imports

- [[generate_peb]]
- [[repo_paths]]

## Imported by

- [[build_comparison_report]]
- [[compare]]
- [[move_and_compare]]
- [[move_pi_to_real]]

## External dependencies

`importlib`, `numpy`, `repo_paths`, `torch`
