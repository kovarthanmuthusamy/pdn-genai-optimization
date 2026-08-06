---
title: latent_traversal
type: code
path: pipelines/analysis/latent_traversal.py
group: pipelines/analysis
loc: 545
tags: [code, pipelines, runnable]
---

# latent_traversal

> Systematic latent-space traversal for Multi-Input VAE.

**Source:** `pipelines/analysis/latent_traversal.py` · 545 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python pipelines/analysis/latent_traversal.py`

## Purpose

```text
Systematic latent-space traversal for Multi-Input VAE.

Run: python pipelines/analysis/latent_traversal.py
```

## Constants

| Name | Value |
|------|-------|
| `PROJECT_ROOT` | `Path(__file__).resolve().parent` |
| `SOURCE_DIR` | `PROJECT_ROOT / 'source'` |
| `VISUALIZATION_DIR` | `PROJECT_ROOT / 'visualization'` |
| `CHECKPOINT_PATH` | `'experiments/exp012/checkpoints/epoch_150.pt'` |
| `LATENT_DIM` | `32` |
| `DEVICE` | `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` |
| `TRAVERSAL_RANGE` | `(-2.0, 2.0)` |
| `NUM_STEPS` | `11` |
| `DIMENSIONS_TO_ANALYZE` | `[0, 1, 2, 3, 4, 5, 8, 10, 15, 20, 25, 31]` |
| `OUTPUT_DIR` | `Path('temp_visuals/latent_traversal')` |
| `STATS_PATH` | `Path('datasets/source/data_norm/normalization_stats.json')` |

## Classes

- **`LatentTraversalAnalyzer`** — Analyzes latent space by systematically varying latent dimensions

## Functions

- **`main()`** — Main execution function

## External dependencies

`matplotlib`, `numpy`, `seaborn`, `source`, `torch`, `tqdm`, `visualization`
