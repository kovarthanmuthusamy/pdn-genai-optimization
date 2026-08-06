---
title: dataloader_multifreq
type: code
path: experiments/exp053_peak_log1p_losses/codes/dataloader_multifreq.py
group: experiments/exp053_peak_log1p_losses/codes
experiment: exp053_peak_log1p_losses
loc: 263
tags: [code, exp053_peak_log1p_losses]
---

# dataloader_multifreq

> exp052 multifreq dataloader — high-MHz cross-freq bias + append-tag curriculum sampling.

**Source:** `experiments/exp053_peak_log1p_losses/codes/dataloader_multifreq.py` · 263 lines
**Experiment:** [[exp053_peak_log1p_losses]]

## Classes

- **`_HighFreqBiasSubset(_base._IndexedSubset)`** — Same-design alt MHz; prefer 300+ MHz targets for cross-freq loss.

## Functions

- **`create_multifreq_data_loaders(data_dir: str, batch_size: int=32, num_workers: int=4, train_split: float=0.9, seed: int=42, pin_memory: bool=True, persistent_workers: bool=False, prefetch_factor: Optional[int]=2, *, split_by_design: bool=True, balance_k: bool=True, balance_freq: bool=True, k_balance_power: float=0.5, freq_balance_power: float=1.0, k_balance_smoothing: float=0.001, stratify_by_k: bool=True, cache_in_ram: bool=False, drop_last_train: bool=True, train_samples_per_epoch: int | None=None, cross_freq_pairs: bool=True, val_batch_size: int | None=None, high_freq_pair_bias: float=0.55, high_freq_mhz_threshold: float=250.0, append_tag_boost_tag: str | None=None, append_tag_boost_frac: float=0.55, append_tag_boost_start_epoch: int=1, append_tag_boost_peak_epoch: int=100, append_tag_boost_decay_end_epoch: int=150)`** — exp038 loader + high-MHz cross-freq bias + optional append-tag curriculum sampler.

## Imports

- [[experiments.exp038_true_multi.codes.dataloader_multifreq]]
- [[experiments.exp053_peak_log1p_losses.codes.sampler_curriculum]]

## Imported by

- [[experiments.exp053_peak_log1p_losses.codes.diagnose_nan_grad]]
- [[experiments.exp053_peak_log1p_losses.codes.train_vae_simple]]

## External dependencies

`numpy`, `torch`
