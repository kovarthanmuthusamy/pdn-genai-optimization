---
title: dataloader_multifreq
type: code
path: experiments/exp043/codes/dataloader_multifreq.py
group: experiments/exp043/codes
experiment: exp043
loc: 382
tags: [code, exp043]
---

# dataloader_multifreq

> Multi-frequency PI heatmap dataloader for exp038_true_multi.

**Source:** `experiments/exp043/codes/dataloader_multifreq.py` · 382 lines
**Experiment:** [[exp043]]

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
- **`_load_pi_freq_mhz(pifreq_dir: Path, stem: str)`**
- **`precompute_multifreq_metadata(dataset: VAEDataset)`** — Cache design keys and frequency bins; written next to dataset root.
- **`_split_indices_by_design(design_keys: list[str], train_split: float, seed: int)`**
- **`_combined_train_weights(train_indices: list[int], design_keys: list[str], freq_bin: list[int], k_values: list[int], *, balance_k: bool, balance_freq: bool, k_power: float, freq_power: float, smooth: float)`**
- **`build_multifreq_pair_lookup(design_keys: list[str], freq_bin: list[int])`** — Map sample index → other indices (same design, different PI_freq bin).
- **`multifreq_collate_fn(batch: list)`**
- **`multifreq_collate_with_alt_fn(batch: list)`** — Stack batches; cross-freq alt fields are prefetched in ``_IndexedSubset``.
- **`make_multifreq_collate_with_pairs(pair_lut: dict[int, list[int]], full_dataset)`** — Legacy collate: alt MHz lookup in collate (slow; used only without RAM prefetch).
- **`create_multifreq_data_loaders(data_dir: str, batch_size: int=32, num_workers: int=4, train_split: float=0.9, seed: int=42, pin_memory: bool=True, persistent_workers: bool=False, prefetch_factor: Optional[int]=2, *, split_by_design: bool=True, balance_k: bool=True, balance_freq: bool=True, k_balance_power: float=0.5, freq_balance_power: float=1.0, k_balance_smoothing: float=0.001, stratify_by_k: bool=True, cache_in_ram: bool=False, drop_last_train: bool=True, train_samples_per_epoch: int | None=None, cross_freq_pairs: bool=True, val_batch_size: int | None=None)`**

## Imports

- [[dataloader]]
- [[multifreq_anchors]]
- [[multifreq_layout_store]]

## Imported by

- [[_audit_decap_locality]]
- [[_audit_physics]]
- [[experiments.exp043.codes.evaluate_vae]]
- [[experiments.exp043.codes.train_vae_simple]]

## External dependencies

`numpy`, `src_vae`, `torch`
