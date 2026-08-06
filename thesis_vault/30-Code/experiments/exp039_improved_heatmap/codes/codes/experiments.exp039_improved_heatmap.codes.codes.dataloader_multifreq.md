---
title: dataloader_multifreq
type: code
path: experiments/exp039_improved_heatmap/codes/codes/dataloader_multifreq.py
group: experiments/exp039_improved_heatmap/codes/codes
experiment: exp039_improved_heatmap
loc: 362
tags: [code, exp039_improved_heatmap]
---

# dataloader_multifreq

> Multi-frequency PI heatmap dataloader for exp038_true_multi.

**Source:** `experiments/exp039_improved_heatmap/codes/codes/dataloader_multifreq.py` · 362 lines
**Experiment:** [[exp039_improved_heatmap]]

## Purpose

```text
Multi-frequency PI heatmap dataloader for exp038_true_multi.

- Train/val split by **design** (same occ+imp = one layout; all MHz stay together).
- Optional **frequency-balanced** + **K-balanced** training sampler.
- Each row: heatmap @ PI_freq, shared occ/imp per design.
```

## Classes

- **`_IndexedSubset(torch.utils.data.Dataset)`** — Wraps a Subset and exposes global dataset indices for pair lookup.

## Functions

- **`_nearest_anchor_mhz(mhz: float)`**
- **`design_key_from_files(occ_path: Path, imp_path: Path)`** — Hash occupancy + impedance ch0 — same layout at all frequencies shares a key.
- **`_load_pi_freq_mhz(pifreq_dir: Path, stem: str)`**
- **`precompute_multifreq_metadata(dataset: VAEDataset)`** — Cache design keys and frequency bins; written next to dataset root.
- **`_split_indices_by_design(design_keys: list[str], train_split: float, seed: int)`**
- **`_combined_train_weights(train_indices: list[int], design_keys: list[str], freq_bin: list[int], k_values: list[int], *, balance_k: bool, balance_freq: bool, k_power: float, freq_power: float, smooth: float)`**
- **`build_multifreq_pair_lookup(design_keys: list[str], freq_bin: list[int])`** — Map sample index → other indices (same design, different PI_freq bin).
- **`multifreq_collate_fn(batch: list)`**
- **`make_multifreq_collate_with_pairs(pair_lut: dict[int, list[int]], full_dataset)`** — Collate that adds cross-frequency heatmap targets from the same layout.
- **`create_multifreq_data_loaders(data_dir: str, batch_size: int=32, num_workers: int=4, train_split: float=0.9, seed: int=42, pin_memory: bool=True, persistent_workers: bool=False, prefetch_factor: Optional[int]=2, *, split_by_design: bool=True, balance_k: bool=True, balance_freq: bool=True, k_balance_power: float=0.5, freq_balance_power: float=1.0, k_balance_smoothing: float=0.001, stratify_by_k: bool=True, cache_in_ram: bool=False, drop_last_train: bool=True, train_samples_per_epoch: int | None=None, cross_freq_pairs: bool=True)`**

## Imports

- [[dataloader]]
- [[multifreq_anchors]]

## External dependencies

`numpy`, `src_vae`, `torch`
