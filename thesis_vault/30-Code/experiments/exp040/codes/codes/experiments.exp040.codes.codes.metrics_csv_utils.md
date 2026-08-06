---
title: metrics_csv_utils
type: code
path: experiments/exp040/codes/codes/metrics_csv_utils.py
group: experiments/exp040/codes/codes
experiment: exp040
loc: 84
tags: [code, exp040]
---

# metrics_csv_utils

> Read/write exp038 metrics CSVs (dedupe by epoch, sorted).

**Source:** `experiments/exp040/codes/codes/metrics_csv_utils.py` · 84 lines
**Experiment:** [[exp040]]

## Functions

- **`dedupe_csv_by_epoch(path: Path, header: list[str], *, epoch_col: str='epoch')`** — Keep last row per epoch; rewrite file sorted. Returns (rows_before, rows_after).
- **`insert_epoch_row(path: Path, header: list[str], row: list)`** — Insert one row; dedupe file first; skip if epoch already exists. Returns True if inserted.
- **`update_metrics_loss_csv(metrics_dir: Path, *, backup: bool=True)`** — Dedupe metrics/loss.csv and optional backup.

## Imports

- [[vae_logger]]

## External dependencies

`src_vae`
