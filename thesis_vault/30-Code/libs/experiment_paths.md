---
title: experiment_paths
type: code
path: libs/experiment_paths.py
group: libs
loc: 62
tags: [code, libs]
---

# experiment_paths

> Shared experiment config loading with repo-relative path resolution.

**Source:** `libs/experiment_paths.py` · 62 lines

## Functions

- **`load_experiment_config(path: Path)`** — Parse experiment config.yaml (JSON object with optional ``#`` comment lines).
- **`data_dir_from_config(config_path: Path, *, fallback: str | Path='data_multi_norm_unbounded')`** — Resolve ``data_dir`` from experiment config under ``REPO_ROOT``.
- **`experiment_dir_from_config(config_path: Path)`** — Resolve ``experiment_dir`` from config, else parent of config file.
- **`norm_stats_path(data_dir: Path)`** — Find normalization_stats.json for a resolved data directory.

## Imports

- [[repo_paths]]

## Imported by

- [[experiments.exp039_improved_heatmap.codes.inference_vae]]
- [[experiments.exp041.codes.inference_vae]]
- [[experiments.exp042.codes.inference_vae]]
- [[experiments.exp043.codes.inference_vae]]
- [[experiments.exp044.codes.inference_vae]]
- [[experiments.exp045.codes.inference_vae]]
- [[experiments.exp046.codes.inference_vae]]
- [[experiments.exp047.codes.inference_vae]]
- [[experiments.exp048.codes.inference_vae]]
- [[experiments.exp049.codes.inference_vae]]
- [[experiments.exp050.codes.inference_vae]]
- [[experiments.exp051_new_datas_appended.codes.inference_vae]]
- [[experiments.exp052_unbounded_pearson.codes.inference_vae]]
- [[experiments.exp053_peak_log1p_losses.codes.inference_vae]]
- [[experiments.exp054_K_30.codes.inference_vae]]
- [[experiments.exp055_hard_occ.codes.inference_vae]]
- [[experiments.exp056_graph_vae.codes.inference_vae]]
- [[experiments.exp057_structured_graph.codes.inference_vae]]
- [[experiments.exp058_asymmetric_kl.codes.inference_vae]]
- [[experiments.exp059_capacity_freq.codes.inference_vae]]
- [[experiments.exp060_multitype_occ.codes.inference_vae]]

## External dependencies

`repo_paths`
