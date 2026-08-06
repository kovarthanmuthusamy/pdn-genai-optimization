---
title: run_vae_eval_heldout
type: code
path: evaluation/vae/run_vae_eval_heldout.py
group: evaluation/vae
loc: 104
tags: [code, evaluation, runnable]
---

# run_vae_eval_heldout

> Full test evaluation on the held-out (non-training) dataset.

**Source:** `evaluation/vae/run_vae_eval_heldout.py` · 104 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python evaluation/vae/run_vae_eval_heldout.py`

## Purpose

```text
Full test evaluation on the held-out (non-training) dataset.

This is the recommended entrypoint to evaluate generalization on combinations
that were *not present in training*.

Assumption
----------
The dataset root you pass is already in VAE-compatible, normalized format:
- heatmap: (1, 64, 64)
- Imp:     (3, 231)
- Occ_map: (52,)

Run
---
    python evaluation/vae/run_vae_eval_heldout.py 
Outputs
-------
Default output directory: evaluation/vae/heldout/
```

## Constants

| Name | Value |
|------|-------|
| `CHECKPOINT` | `Path('experiments/exp027_sigma_reg_tuning/checkpoints/checkpoint_epoch_400.pt')` |
| `DATASET_ROOT` | `Path('datasets/data_eval_norm')` |
| `OUT_DIR` | `Path('evaluation/vae/heldout')` |
| `BATCH_SIZE` | `64` |
| `N_GEN` | `2048` |
| `SHARED_TEMP` | `1.5` |
| `SEED` | `0` |
| `FORCE_CPU` | `False` |

## Functions

- **`_project_root()`**
- **`_dataset_ready(root: Path)`**
- **`_import_run_vae_eval(project_root: Path)`**
- **`main()`**

## External dependencies

`importlib`
