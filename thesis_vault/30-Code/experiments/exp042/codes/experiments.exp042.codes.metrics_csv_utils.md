---
title: metrics_csv_utils
type: code
path: experiments/exp042/codes/metrics_csv_utils.py
group: experiments/exp042/codes
experiment: exp042
loc: 121
tags: [code, exp042]
---

# metrics_csv_utils

> Read/write exp042 metrics CSVs (dedupe by epoch, sorted).

**Source:** `experiments/exp042/codes/metrics_csv_utils.py` · 121 lines
**Experiment:** [[exp042]]

## Purpose

```text
Read/write exp042 metrics CSVs (dedupe by epoch, sorted).

This experiment can produce duplicate epoch rows when training is resumed or a
run restarts mid-epoch. These helpers rewrite CSVs keeping the *last* row per
epoch and sorting by epoch.
```

## Constants

| Name | Value |
|------|-------|
| `EPOCH_TIMING_HEADER` | `['epoch', 'train_sec', 'val_sec', 'total_sec', 'val_ran', 'train_loss', 'val_loss']` |

## Functions

- **`dedupe_csv_by_epoch(path: Path, header: list[str], *, epoch_col: str='epoch')`** — Keep last row per epoch; rewrite file sorted. Returns (rows_before, rows_after).
- **`insert_epoch_row(path: Path, header: list[str], row: list)`** — Insert one row; dedupe file first; skip if epoch already exists. Returns True if inserted.
- **`update_metrics_loss_csv(metrics_dir: Path, *, backup: bool=True)`** — Dedupe metrics/loss.csv and optional backup.
- **`update_epoch_timing_csv(metrics_dir: Path, *, backup: bool=True)`** — Dedupe metrics/epoch_timing.csv and optional backup.
- **`update_all_metrics_csv(metrics_dir: Path, *, backup: bool=True)`** — Dedupe all exp042 metrics CSVs that are epoch-indexed.

## Imports

- [[vae_logger]]

## External dependencies

`src_vae`
