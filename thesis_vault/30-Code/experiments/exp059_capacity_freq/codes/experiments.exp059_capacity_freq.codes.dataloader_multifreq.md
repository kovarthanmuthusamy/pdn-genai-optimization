---
title: dataloader_multifreq
type: code
path: experiments/exp059_capacity_freq/codes/dataloader_multifreq.py
group: experiments/exp059_capacity_freq/codes
experiment: exp059_capacity_freq
loc: 196
tags: [code, exp059_capacity_freq, uncommitted]
---

# dataloader_multifreq

> exp055 multifreq dataloader — cross-freq pairs + optional K/freq balance.

**Source:** `experiments/exp059_capacity_freq/codes/dataloader_multifreq.py` · 196 lines
**Git:** uncommitted — not yet tracked
**Experiment:** [[exp059_capacity_freq]]

## Functions

- **`create_multifreq_data_loaders(data_dir: str, batch_size: int=32, num_workers: int=4, train_split: float=0.9, seed: int=42, pin_memory: bool=True, persistent_workers: bool=False, prefetch_factor: Optional[int]=2, *, split_by_design: bool=True, balance_k: bool=True, balance_freq: bool=True, k_balance_power: float=0.5, freq_balance_power: float=1.0, k_balance_smoothing: float=0.001, stratify_by_k: bool=True, cache_in_ram: bool=False, drop_last_train: bool=True, train_samples_per_epoch: int | None=None, cross_freq_pairs: bool=True, val_batch_size: int | None=None, ddp_rank: int=0, ddp_world_size: int=1, al_overlay_data_dir: str | None=None, al_overlay_sample_weight: float=25.0)`**

## Imports

- [[experiments.exp059_capacity_freq.codes.dataloader_base]]

## Imported by

- [[experiments.exp059_capacity_freq.codes.train_core]]
- [[experiments.exp059_capacity_freq.codes.train_vae_simple]]

## External dependencies

`torch`
