---
title: latent_traversal
type: code
path: evaluation/vae/latent_traversal.py
group: evaluation/vae
loc: 671
tags: [code, evaluation, runnable]
---

# latent_traversal

> Latent Traversal Analysis for Multi-Input VAE

**Source:** `evaluation/vae/latent_traversal.py` · 671 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python evaluation/vae/latent_traversal.py`

## Purpose

```text
Latent Traversal Analysis for Multi-Input VAE
=====================================

This script performs systematic latent space exploration to understand which latent 
dimensions control which modalities (heatmap, occupancy, impedance).

Key Analysis:
- Sweeps selected latent dimensions from -2 to +2
- Generates samples at each point
- Visualizes how each modality responds to changes
- Identifies which latent neurons are most influential for each output

Usage:
    python latent_traversal.py
    
Output:
    - Individual traversal plots for each latent dimension
    - Response analysis showing modality sensitivity
    - Combined dashboard showing all traversals
```

## Constants

| Name | Value |
|------|-------|
| `REPO_ROOT` | `Path(__file__).resolve().parents[2]` |
| `VISUALIZATION_DIR` | `REPO_ROOT / 'evaluation' / 'visualization'` |
| `CHECKPOINT_PATH` | `'experiments/exp028_eval_improvement/checkpoints/checkpoint_epoch_400.pt'` |
| `LATENT_DIM` | `32` |
| `DEVICE` | `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` |
| `TRAVERSAL_RANGE` | `(-2.0, 2.0)` |
| `NUM_STEPS` | `11` |
| `DIMENSIONS_TO_ANALYZE` | `[0, 1, 2, 3, 4, 5, 8, 10, 15, 20, 25, 31]` |
| `K_VALUE` | `5` |
| `OUTPUT_DIR` | `Path('evaluation/latent_traversal')` |
| `STATS_PATH` | `Path('datasets/data_norm/normalization_stats.json')` |

## Classes

- **`LatentTraversalAnalyzer`** — Analyzes latent space by systematically varying latent dimensions

## Functions

- **`main()`** — Main execution function

## External dependencies

`experiments`, `matplotlib`, `numpy`, `seaborn`, `torch`, `tqdm`, `visualization`
