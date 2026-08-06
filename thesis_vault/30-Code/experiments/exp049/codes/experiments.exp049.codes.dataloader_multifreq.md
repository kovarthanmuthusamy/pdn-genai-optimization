---
title: dataloader_multifreq
type: code
path: experiments/exp049/codes/dataloader_multifreq.py
group: experiments/exp049/codes
experiment: exp049
loc: 213
tags: [code, exp049]
---

# dataloader_multifreq

> exp049 multifreq dataloader — bias cross-freq alt pairs toward high MHz.

**Source:** `experiments/exp049/codes/dataloader_multifreq.py` · 213 lines
**Experiment:** [[exp049]]

## Classes

- **`_HighFreqBiasSubset(_base._IndexedSubset)`** — Same-design alt MHz; prefer 300+ MHz targets for cross-freq loss.

## Functions

- **`create_multifreq_data_loaders(data_dir: str, batch_size: int=32, num_workers: int=4, train_split: float=0.9, seed: int=42, pin_memory: bool=True, persistent_workers: bool=False, prefetch_factor: Optional[int]=2, *, split_by_design: bool=True, balance_k: bool=True, balance_freq: bool=True, k_balance_power: float=0.5, freq_balance_power: float=1.0, k_balance_smoothing: float=0.001, stratify_by_k: bool=True, cache_in_ram: bool=False, drop_last_train: bool=True, train_samples_per_epoch: int | None=None, cross_freq_pairs: bool=True, val_batch_size: int | None=None, high_freq_pair_bias: float=0.55, high_freq_mhz_threshold: float=250.0)`** — Same as exp038 loader but with high-MHz cross-freq pair bias on train.

## Imports

- [[experiments.exp038_true_multi.codes.dataloader_multifreq]]

## Imported by

- [[experiments.exp049.codes.train_vae_simple]]

## External dependencies

`numpy`, `torch`
