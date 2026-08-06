---
title: dataloader
type: code
path: src_vae/others/dataloader.py
group: src_vae/others
loc: 420
tags: [code, src_vae]
---

# dataloader

> PyTorch dataset and data loaders for multi-input VAE training.

**Source:** `src_vae/others/dataloader.py` · 420 lines

## Purpose

```text
PyTorch dataset and data loaders for multi-input VAE training.

Purpose: Load heatmap, occupancy, impedance, and optional PI_freq tensors from
    layout-centric multifreq or legacy single-freq on-disk layouts.
Run: ``from src_vae.others.dataloader import create_data_loaders`` in training scripts.
Inputs / outputs: Dataset dirs (``data_multifreq*``, ``data_norm``); yields dicts with
    ``heatmap_norm``, ``occupancy``, ``impedance``, ``K``, ``PI_freq``, ``filename``.
Dependencies: ``torch``, ``numpy``; ``heatmap_z_clip``, ``multifreq_layout_store``.
Agent notes:
    - Type: library module (import-only).
    - Key symbols: ``VAEDataset``, ``create_data_loaders``, ``collate_fn``.
    - Multifreq layout: ``heatmap/sample_N.npy``, ``PI_freq/sample_N.npy`` per row;
      ``layouts/{design_id}/imp.npy``, ``occ.npy`` via ``manifest.csv``.
```

## Classes

- **`VAEDataset(Dataset)`** — PyTorch Dataset: heatmap + occupancy + impedance (+ optional PI_freq).

## Functions

- **`create_data_loaders(data_dir: str='datasets/data_norm', batch_size: int=32, num_workers: int=4, normalize: bool=False, stats_path: Optional[str]=None, train_split: float=0.8, seed: int=42, pin_memory: bool=True, persistent_workers: bool=False, prefetch_factor: Optional[int]=2, *, stratify_by_k: bool=False, balance_k: bool=False, k_balance_power: float=1.0, k_balance_smoothing: float=0.001, k_balance_print: bool=True, cache_in_ram: bool=False, drop_last_train: bool=True)`** — Create train and validation data loaders.
- **`collate_fn(batch: list)`**

## Imports

- [[heatmap_z_clip]]
- [[multifreq_layout_store]]
- [[norm_stats]]
- [[pi_freq_utils]]

## Imported by

- [[exp041_eval_common]]
- [[exp042_eval_common]]
- [[exp043_eval_common]]
- [[exp044_eval_common]]
- [[exp045_eval_common]]
- [[exp046_eval_common]]
- [[exp047_eval_common]]
- [[exp048_eval_common]]
- [[exp049_eval_common]]
- [[exp050_eval_common]]
- [[exp051_eval_common]]
- [[exp052_eval_common]]
- [[exp053_eval_common]]
- [[experiments.exp037_lat_change.codes.evaluate_vae]]
- [[experiments.exp037_lat_change.codes.train_vae_simple]]
- [[experiments.exp037_lat_change.codes.visualize_latent]]
- [[experiments.exp038_true_multi.codes.dataloader_multifreq]]
- [[experiments.exp038_true_multi.codes.evaluate_vae]]
- [[experiments.exp038_true_multi.codes.visualize_latent]]
- [[experiments.exp039_improved_heatmap.codes.codes.dataloader_multifreq]]
- [[experiments.exp039_improved_heatmap.codes.codes.evaluate_vae]]
- [[experiments.exp039_improved_heatmap.codes.codes.visualize_latent]]
- [[experiments.exp039_improved_heatmap.codes.evaluate_vae]]
- [[experiments.exp039_improved_heatmap.codes.exp039_eval_common]]
- [[experiments.exp040.codes.codes.dataloader_multifreq]]
- [[experiments.exp040.codes.codes.evaluate_vae]]
- [[experiments.exp040.codes.codes.visualize_latent]]
- [[experiments.exp040.codes.evaluate_vae]]
- [[experiments.exp040.codes.exp039_eval_common]]
- [[experiments.exp041.codes.codes.dataloader_multifreq]]
- [[experiments.exp041.codes.codes.evaluate_vae]]
- [[experiments.exp041.codes.codes.visualize_latent]]
- [[experiments.exp041.codes.evaluate_vae]]
- [[experiments.exp042.codes.evaluate_vae]]
- [[experiments.exp043.codes.dataloader_multifreq]]
- [[experiments.exp043.codes.evaluate_vae]]
- [[experiments.exp044.codes.evaluate_vae]]
- [[experiments.exp044.codes.latent_optimization_joint]]
- [[experiments.exp045.codes.evaluate_vae]]
- [[experiments.exp045.codes.latent_optimization_joint]]
- [[experiments.exp054_K_30.codes.dataloader_base]]
- [[experiments.exp055_hard_occ.codes.dataloader_base]]
- [[experiments.exp056_graph_vae.codes.dataloader_base]]
- [[experiments.exp057_structured_graph.codes.dataloader_base]]
- [[experiments.exp058_asymmetric_kl.codes.dataloader_base]]
- [[experiments.exp059_capacity_freq.codes.dataloader_base]]
- [[experiments.exp060_multitype_occ.codes.dataloader_base]]

## External dependencies

`numpy`, `torch`
