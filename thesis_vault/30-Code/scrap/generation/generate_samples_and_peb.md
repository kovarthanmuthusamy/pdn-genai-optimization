---
title: generate_samples_and_peb
type: code
path: scrap/generation/generate_samples_and_peb.py
group: scrap/generation
loc: 180
tags: [code, scrap, runnable]
---

# generate_samples_and_peb

> Generate VAE samples and PEB for one K.

**Source:** `scrap/generation/generate_samples_and_peb.py` · 180 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python scrap/generation/generate_samples_and_peb.py`

## Purpose

```text
Generate VAE samples and PEB for one K.

Purpose:
    Load VAE checkpoint, generate N samples with exactly K active decaps, save ``data_sample_*``
    folders, and write a matching ECADStar ``.peb``.

Run:
    python scrap/generation/generate_samples_and_peb.py

Agent notes:
    - What: Single-K VAE sample export + PEB builder (legacy exp030 path).
    - Usage: Set ``K_VALUE``, ``NUM_SAMPLES``, checkpoint paths, ``OUTPUT_DIR`` → run.
    - Config keys:
        - ``CHECKPOINT_PATH``, ``LATENT_STATS_PATH``, ``MODEL_LATENT_DIM`` — VAE load
        - ``K_VALUE``, ``NUM_SAMPLES``, ``SHARED_TEMP`` — generation
        - ``OUTPUT_DIR``, ``PEB_PATH`` — where ``.npy`` and ``.peb`` are written
    - Key symbols: ``generate_save``, ``main``
```

## Constants

| Name | Value |
|------|-------|
| `CHECKPOINT_PATH` | `'experiments/exp030_adding_physic/checkpoints/checkpoint_epoch_400.pt'` |
| `LATENT_STATS_PATH` | `''` |
| `MODEL_LATENT_DIM` | `32` |
| `NUM_SAMPLES` | `5` |
| `K_VALUE` | `5` |
| `SHARED_TEMP` | `1.5` |
| `OUTPUT_DIR` | `f'scrap/generated_samples_v2/K{K_VALUE}'` |
| `PEB_PATH` | `'scrap/PEB'` |
| `POWERBUS` | `'Power_GND'` |
| `FREQ` | `'63e6'` |
| `COMPONENTS` | `'IC1_Port1'` |
| `FORCE_CPU` | `False` |
| `PROJECT_ROOT` | `_add_project_root_to_syspath()` |

## Functions

- **`_add_project_root_to_syspath()`**
- **`generate_save(engine: VAEInference, *, num_samples: int, out_dir: str | Path, K: int, shared_temp: float)`** — Generate samples and save them to disk.
- **`main()`**

## Imports

- [[generate_peb]]
- [[repo_paths]]

## Imported by

- [[generate_samples_and_peb_all_k]]

## External dependencies

`experiments`, `numpy`, `repo_paths`, `torch`
