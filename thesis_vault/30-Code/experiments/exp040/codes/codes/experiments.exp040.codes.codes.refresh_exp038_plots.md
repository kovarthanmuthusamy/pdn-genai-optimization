---
title: refresh_exp038_plots
type: code
path: experiments/exp040/codes/codes/refresh_exp038_plots.py
group: experiments/exp040/codes/codes
experiment: exp040
loc: 70
tags: [code, exp040]
---

# refresh_exp038_plots

> Update metrics/loss.csv (dedupe) and rebuild exp038 plots from VAETrainingLogger.

**Source:** `experiments/exp040/codes/codes/refresh_exp038_plots.py` · 70 lines
**Experiment:** [[exp040]]

## Purpose

```text
Update metrics/loss.csv (dedupe) and rebuild exp038 plots from VAETrainingLogger.

    python experiments/exp038_true_multi/codes/refresh_exp038_plots.py

Skips epoch-1 GPU backfill; run save_epoch1_losses.py first if plots should start at ep 1.
```

## Constants

| Name | Value |
|------|-------|
| `_ROOT` | `Path(__file__).resolve()` |
| `PROJECT_ROOT` | `next((str(p) for p in _ROOT.parents if (p / 'src_vae').is_dir()), str(_ROOT.parents[3]))` |

## Functions

- **`main()`**

## Imports

- [[experiments.exp038_true_multi.codes.metrics_csv_utils]]
- [[experiments.exp038_true_multi.codes.train_vae_simple]]
- [[vae_logger]]

## External dependencies

`src_vae`
