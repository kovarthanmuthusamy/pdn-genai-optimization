---
title: save_epoch1_losses
type: code
path: experiments/exp040/codes/codes/save_epoch1_losses.py
group: experiments/exp040/codes/codes
experiment: exp040
loc: 186
tags: [code, exp040]
---

# save_epoch1_losses

> Backfill epoch-1 train/val metrics into metrics/loss.csv (no checkpoint write).

**Source:** `experiments/exp040/codes/codes/save_epoch1_losses.py` · 186 lines
**Experiment:** [[exp040]]

## Purpose

```text
Backfill epoch-1 train/val metrics into metrics/loss.csv (no checkpoint write).

Replays one training epoch with fresh init + seed=42 (same as a new exp038 run at ep 1).

    python experiments/exp038_true_multi/codes/save_epoch1_losses.py
    python experiments/exp038_true_multi/codes/refresh_exp038_plots.py
```

## Constants

| Name | Value |
|------|-------|
| `_ROOT` | `Path(__file__).resolve()` |
| `PROJECT_ROOT` | `next((str(p) for p in _ROOT.parents if (p / 'src_vae').is_dir()), str(_ROOT.parents[3]))` |
| `_BACKFILL_RAM` | `os.getenv('EXP038_BACKFILL_RAM', '0').strip().lower() in ('1', 'true', 'yes')` |

## Functions

- **`_seed_all(seed: int=42)`**
- **`main()`**

## Imports

- [[experiments.exp038_true_multi.codes.dataloader_multifreq]]
- [[experiments.exp038_true_multi.codes.metrics_csv_utils]]
- [[experiments.exp038_true_multi.codes.physics_loss]]
- [[experiments.exp038_true_multi.codes.train_vae_simple]]
- [[experiments.exp038_true_multi.codes.vae_multi_input_simple]]
- [[vae_logger]]

## External dependencies

`numpy`, `src_vae`, `torch`
