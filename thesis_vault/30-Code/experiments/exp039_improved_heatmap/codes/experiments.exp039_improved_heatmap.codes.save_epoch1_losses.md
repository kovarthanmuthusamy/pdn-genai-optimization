---
title: save_epoch1_losses
type: code
path: experiments/exp039_improved_heatmap/codes/save_epoch1_losses.py
group: experiments/exp039_improved_heatmap/codes
experiment: exp039_improved_heatmap
loc: 265
tags: [code, exp039_improved_heatmap, runnable]
---

# save_epoch1_losses

> Backfill epoch-1 metrics and rebuild exp039 plots from metrics/loss.csv.

**Source:** `experiments/exp039_improved_heatmap/codes/save_epoch1_losses.py` · 265 lines
**Experiment:** [[exp039_improved_heatmap]]
**Runnable:** CONFIG-only script — edit constants at top, then `python experiments/exp039_improved_heatmap/codes/save_epoch1_losses.py`

## Purpose

```text
Backfill epoch-1 metrics and rebuild exp039 plots from metrics/loss.csv.

Use when a run was resumed from a later checkpoint (e.g. 750) so loss.csv starts
at epoch 25 and final plots omit epoch 1.

  cd ~/gan
  # Backfill epoch 1 (one GPU epoch, fresh init) + rebuild plots:
  python experiments/exp039_improved_heatmap/codes/save_epoch1_losses.py

  # Only dedupe CSV and rebuild plots (epoch 1 already in loss.csv):

Writes / updates:
  - metrics/loss.csv  (epoch=1 row when backfilled)
  - logs/latent_stats.csv  (epoch=1, when backfilled)
  - metrics/plots/*_final.png  (full curve from CSV, including ep 1 if present)
```

## Constants

| Name | Value |
|------|-------|
| `_ROOT` | `Path(__file__).resolve()` |
| `PROJECT_ROOT` | `next((str(p) for p in _ROOT.parents if (p / 'src_vae').is_dir()), str(_ROOT.parents[3]))` |
| `_EXP_DIR` | `_ROOT.parents[1]` |
| `PLOTS_ONLY` | `False` |
| `_BACKFILL_RAM` | `os.getenv('EXP039_BACKFILL_RAM', '0').strip().lower() in ('1', 'true', 'yes')` |

## Functions

- **`_seed_all(seed: int=42)`**
- **`rebuild_plots_from_loss_csv(c: Config)`** — Reload deduped loss.csv and write the same *_final.png plots as end of training.
- **`backfill_epoch1(c: Config)`** — Run one training epoch (fresh weights) and insert epoch=1 into metrics CSVs.
- **`main()`**

## Imports

- [[experiments.exp038_true_multi.codes.dataloader_multifreq]]
- [[experiments.exp038_true_multi.codes.physics_loss]]
- [[experiments.exp038_true_multi.codes.train_vae_simple]]
- [[experiments.exp038_true_multi.codes.vae_multi_input_simple]]
- [[experiments.exp039_improved_heatmap.codes.metrics_csv_utils]]
- [[vae_logger]]

## External dependencies

`numpy`, `src_vae`, `torch`
