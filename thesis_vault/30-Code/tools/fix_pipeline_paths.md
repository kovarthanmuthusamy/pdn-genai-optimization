---
title: fix_pipeline_paths
type: code
path: tools/fix_pipeline_paths.py
group: tools
loc: 85
tags: [code, tools]
---

# fix_pipeline_paths

> Fix repo-root path depth after pipelines/ migration.

**Source:** `tools/fix_pipeline_paths.py` · 85 lines

## Constants

| Name | Value |
|------|-------|
| `ROOT` | `Path(__file__).resolve().parents[1]` |
| `PIPELINES` | `ROOT / 'pipelines'` |
| `REPLACEMENTS` | `[('Path\\(__file__\\)\\.resolve\\(\\)\\.parents\\[1\\]', 'Path(__file__).resolve().parent…` |
| `OPT_LOADER` | `PIPELINES / 'latent' / 'optimization_loader.py'` |
| `FILTER_COMBO` | `PIPELINES / 'heatmaps' / 'filter_combinations.py'` |
| `HEATMAP_CHANGE` | `PIPELINES / 'heatmaps' / 'change_frequency.py'` |
| `HEATMAP_REGEN` | `PIPELINES / 'heatmaps' / 'regenerate_mhz_pebs.py'` |
