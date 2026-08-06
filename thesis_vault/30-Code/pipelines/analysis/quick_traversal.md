---
title: quick_traversal
type: code
path: pipelines/analysis/quick_traversal.py
group: pipelines/analysis
loc: 82
tags: [code, pipelines, runnable]
---

# quick_traversal

> Quick latent-dimension traversal smoke test.

**Source:** `pipelines/analysis/quick_traversal.py` · 82 lines
**Runnable:** CONFIG-only script — edit constants at top, then `python pipelines/analysis/quick_traversal.py`

## Purpose

```text
Quick latent-dimension traversal smoke test.

Run: python pipelines/analysis/quick_traversal.py
```

## Constants

| Name | Value |
|------|-------|
| `SOURCE_DIR` | `repo_path('source')` |
| `CHECKPOINT_PATH` | `repo_path('experiments/exp012/checkpoints/epoch_100.pt')` |
| `LATENT_DIM` | `32` |
| `DEVICE` | `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` |
| `OUTPUT_DIR` | `repo_path('temp_visuals/quick_traversal_test')` |
| `TEST_DIMENSION` | `0` |
| `TRAVERSAL_VALUES` | `np.linspace(-2, 2, 5)` |

## Functions

- **`main()`**

## Imports

- [[repo_paths]]

## External dependencies

`matplotlib`, `numpy`, `repo_paths`, `source`, `torch`
