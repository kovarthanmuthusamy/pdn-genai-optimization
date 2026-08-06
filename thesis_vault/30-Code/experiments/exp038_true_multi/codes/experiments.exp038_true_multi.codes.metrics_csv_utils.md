---
title: metrics_csv_utils
type: code
path: experiments/exp038_true_multi/codes/metrics_csv_utils.py
group: experiments/exp038_true_multi/codes
experiment: exp038_true_multi
loc: 84
tags: [code, exp038_true_multi]
---

# metrics_csv_utils

> Read/write exp038 metrics CSVs (dedupe by epoch, sorted).

**Source:** `experiments/exp038_true_multi/codes/metrics_csv_utils.py` · 84 lines
**Experiment:** [[exp038_true_multi]]

## Functions

- **`dedupe_csv_by_epoch(path: Path, header: list[str], *, epoch_col: str='epoch')`** — Keep last row per epoch; rewrite file sorted. Returns (rows_before, rows_after).
- **`insert_epoch_row(path: Path, header: list[str], row: list)`** — Insert one row; dedupe file first; skip if epoch already exists. Returns True if inserted.
- **`update_metrics_loss_csv(metrics_dir: Path, *, backup: bool=True)`** — Dedupe metrics/loss.csv and optional backup.

## Imports

- [[vae_logger]]

## Imported by

- [[experiments.exp038_true_multi.codes.refresh_exp038_plots]]
- [[experiments.exp038_true_multi.codes.save_epoch1_losses]]
- [[experiments.exp039_improved_heatmap.codes.codes.refresh_exp038_plots]]
- [[experiments.exp039_improved_heatmap.codes.codes.save_epoch1_losses]]
- [[experiments.exp040.codes.codes.refresh_exp038_plots]]
- [[experiments.exp040.codes.codes.save_epoch1_losses]]
- [[experiments.exp041.codes.codes.refresh_exp038_plots]]
- [[experiments.exp041.codes.codes.save_epoch1_losses]]

## External dependencies

`src_vae`
