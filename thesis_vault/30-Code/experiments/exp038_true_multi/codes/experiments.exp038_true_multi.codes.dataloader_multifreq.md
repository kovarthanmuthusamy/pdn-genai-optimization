---
title: dataloader_multifreq
type: code
path: experiments/exp038_true_multi/codes/dataloader_multifreq.py
group: experiments/exp038_true_multi/codes
experiment: exp038_true_multi
loc: 389
tags: [code, exp038_true_multi]
---

# dataloader_multifreq

> Multi-frequency PI heatmap dataloader for exp038_true_multi.

**Source:** `experiments/exp038_true_multi/codes/dataloader_multifreq.py` · 389 lines
**Experiment:** [[exp038_true_multi]]

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

- **`_anchors_for_dataset(data_dir: Path | str)`**
- **`_nearest_anchor_mhz(mhz: float, anchors: tuple[float, ...])`**
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

- [[check_exp044_norm_vs_train]]
- [[debug_clip_spatial]]
- [[debug_inspect_robust_stats]]
- [[diagnose_exp044_sweep]]
- [[diagnose_sweep_vs_dataset]]
- [[experiments.exp038_true_multi.codes.compute_anchor_fg_max]]
- [[experiments.exp038_true_multi.codes.eval_cross_freq]]
- [[experiments.exp038_true_multi.codes.eval_val_recon]]
- [[experiments.exp038_true_multi.codes.freq_inference_utils]]
- [[experiments.exp038_true_multi.codes.save_epoch1_losses]]
- [[experiments.exp038_true_multi.codes.train_vae_simple]]
- [[experiments.exp039_improved_heatmap.codes.codes.compute_anchor_fg_max]]
- [[experiments.exp039_improved_heatmap.codes.codes.eval_cross_freq]]
- [[experiments.exp039_improved_heatmap.codes.codes.eval_val_recon]]
- [[experiments.exp039_improved_heatmap.codes.codes.freq_inference_utils]]
- [[experiments.exp039_improved_heatmap.codes.codes.save_epoch1_losses]]
- [[experiments.exp039_improved_heatmap.codes.codes.train_vae_simple]]
- [[experiments.exp039_improved_heatmap.codes.evaluate_vae]]
- [[experiments.exp039_improved_heatmap.codes.save_epoch1_losses]]
- [[experiments.exp040.codes.codes.compute_anchor_fg_max]]
- [[experiments.exp040.codes.codes.eval_cross_freq]]
- [[experiments.exp040.codes.codes.eval_val_recon]]
- [[experiments.exp040.codes.codes.freq_inference_utils]]
- [[experiments.exp040.codes.codes.save_epoch1_losses]]
- [[experiments.exp040.codes.codes.train_vae_simple]]
- [[experiments.exp040.codes.evaluate_vae]]
- [[experiments.exp040.codes.save_epoch1_losses]]
- [[experiments.exp041.codes.codes.compute_anchor_fg_max]]
- [[experiments.exp041.codes.codes.eval_cross_freq]]
- [[experiments.exp041.codes.codes.eval_val_recon]]
- [[experiments.exp041.codes.codes.freq_inference_utils]]
- [[experiments.exp041.codes.codes.save_epoch1_losses]]
- [[experiments.exp041.codes.codes.train_vae_simple]]
- [[experiments.exp041.codes.evaluate_vae]]
- [[experiments.exp041.codes.save_epoch1_losses]]
- [[experiments.exp042.codes.evaluate_vae]]
- [[experiments.exp043.codes.freq_inference_utils]]
- [[experiments.exp044.codes.evaluate_vae]]
- [[experiments.exp045.codes.evaluate_vae]]
- [[experiments.exp046.codes.eval_real_data_sweep]]
- [[experiments.exp046.codes.eval_train_vs_val]]
- [[experiments.exp047.codes.dataloader_multifreq]]
- [[experiments.exp047.codes.eval_real_data_sweep]]
- [[experiments.exp047.codes.eval_train_vs_val]]
- [[experiments.exp048.codes.dataloader_multifreq]]
- [[experiments.exp048.codes.eval_real_data_sweep]]
- [[experiments.exp048.codes.eval_train_vs_val]]
- [[experiments.exp049.codes.dataloader_multifreq]]
- [[experiments.exp049.codes.eval_real_data_sweep]]
- [[experiments.exp049.codes.eval_train_vs_val]]
- [[experiments.exp050.codes.dataloader_multifreq]]
- [[experiments.exp050.codes.eval_real_data_sweep]]
- [[experiments.exp050.codes.eval_train_vs_val]]
- [[experiments.exp051_new_datas_appended.codes.dataloader_multifreq]]
- [[experiments.exp051_new_datas_appended.codes.eval_real_data_sweep]]
- [[experiments.exp051_new_datas_appended.codes.eval_train_vs_val]]
- [[experiments.exp052_unbounded_pearson.codes.dataloader_multifreq]]
- [[experiments.exp052_unbounded_pearson.codes.eval_real_data_sweep]]
- [[experiments.exp052_unbounded_pearson.codes.eval_train_vs_val]]
- [[experiments.exp053_peak_log1p_losses.codes.dataloader_multifreq]]
- [[experiments.exp053_peak_log1p_losses.codes.eval_real_data_sweep]]
- [[experiments.exp053_peak_log1p_losses.codes.eval_train_vs_val]]
- [[run_multifreq_heatmap_sweep]]
- [[sweep_latent_opt_rules]]
- [[sweep_qc_eval]]

## External dependencies

`numpy`, `src_vae`, `torch`
