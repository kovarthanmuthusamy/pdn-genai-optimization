# Pipelines

Runnable workflow scripts for dataset building, normalization, latent optimization, visualization, and active learning.

## How to run

Every script here is **CONFIG-only**:

1. Open the script.
2. Edit the block marked `# CONFIGURATION — edit these before running`.
3. Run with no extra arguments:

```bash
python pipelines/<folder>/<script>.py
```

Shared JSON settings for active learning live in `active_learning_pi/config/default.json` (see `pipelines/active_learning/run.py`).

## Script documentation

Every pipeline script has a module docstring with **Purpose**, **Run**, and **Agent notes**:

- **What** — what the script does in one sentence
- **Usage** — edit the CONFIG block, then `python <path>`
- **Config keys** — each constant and its meaning

Library code under `libs/` is import-only (no CONFIG block).

## Layout

| Folder | Purpose |
|--------|---------|
| `data/` | Build datasets from raw ECAD exports (multifreq, eval, single-freq legacy) |
| `dataset/` | Manifest transforms, subsample by K, gmax rebuild, extract subset |
| `normalize/` | Normalization, stats, layout-store verification |
| `analysis/` | Latent traversal, mask checks, QA utilities |
| `visualize/` | Heatmap and impedance plotting |
| `latent/` | Stage-2 optimization, PEB export, comparison reports |
| `heatmaps/` | PEB frequency tools and combination filtering |
| `active_learning/` | Active-learning cycle entry (`run.py`) |

## Common commands

```bash
python pipelines/data/processing_multifreq.py
python pipelines/normalize/multifreq.py
python pipelines/latent/optimize.py
python pipelines/active_learning/run.py
```

## Libraries vs pipelines

Import-only helpers live under `libs/` (`data_creation`, `peb`, `dataset_meta`). Pipelines import those modules; they are not run directly unless noted in their docstring.

See also: [`../docs/data-pipeline.md`](../docs/data-pipeline.md), [`../docs/README.md`](../docs/README.md).
