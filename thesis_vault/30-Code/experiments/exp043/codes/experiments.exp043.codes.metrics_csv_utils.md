---
title: metrics_csv_utils
type: code
path: experiments/exp043/codes/metrics_csv_utils.py
group: experiments/exp043/codes
experiment: exp043
loc: 128
tags: [code, exp043]
---

# metrics_csv_utils

> Dedupe and sort exp043 training metrics CSVs by epoch.

**Source:** `experiments/exp043/codes/metrics_csv_utils.py` · 128 lines
**Experiment:** [[exp043]]

## Purpose

```text
Dedupe and sort exp043 training metrics CSVs by epoch.

Purpose:
    Remove duplicate epoch rows after training resume; keep last row per epoch, sorted.

Run:
    Import only — ``from experiments.exp043.codes.metrics_csv_utils import update_all_metrics_csv``.

Agent notes:
    - What: Post-training CSV hygiene for ``metrics/loss.csv`` and timing files after checkpoint resume.
    - Usage: Call ``update_all_metrics_csv(metrics_dir)`` after interrupted training or before replotting.
    - Key symbols: ``dedupe_csv_by_epoch``, ``update_all_metrics_csv``, ``insert_epoch_row``
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
