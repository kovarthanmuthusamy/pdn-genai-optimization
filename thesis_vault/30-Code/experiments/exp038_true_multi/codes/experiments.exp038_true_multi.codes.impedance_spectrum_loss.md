---
title: impedance_spectrum_loss
type: code
path: experiments/exp038_true_multi/codes/impedance_spectrum_loss.py
group: experiments/exp038_true_multi/codes
experiment: exp038_true_multi
loc: 205
tags: [code, exp038_true_multi]
---

# impedance_spectrum_loss

> Peak-aware impedance losses (frequency weighting, dual top-k, peak alignment).

**Source:** `experiments/exp038_true_multi/codes/impedance_spectrum_loss.py` · 205 lines
**Experiment:** [[exp038_true_multi]]

## Classes

- **`ImpedanceSpectrumWeights`**

## Functions

- **`imp_ch0(x: torch.Tensor)`**
- **`resonance_freq_weights(target: torch.Tensor, alpha: float)`** — Up-weight bins where target log-Z has strong negative curvature (resonances).
- **`local_peak_indices(x: torch.Tensor, num_peaks: int)`** — Strongest local maxima indices per row, shape (B, num_peaks).
- **`dual_topk_loss(recon: torch.Tensor, target: torch.Tensor, k: int, under_penalty: float=1.0)`** — Top-k on target bins plus top-k on recon bins (catches misplaced peaks).
- **`_greedy_peak_index_loss(ti_n: torch.Tensor, ri_n: torch.Tensor, *, delta: float=0.08)`** — Match each target peak to nearest unused recon peak (normalized frequency).
- **`peak_alignment_loss(recon: torch.Tensor, target: torch.Tensor, num_peaks: int)`** — Penalize peak position drift (greedy match) and magnitude at target peaks.
- **`impedance_spectrum_loss(recon: torch.Tensor, target: torch.Tensor, *, imp_log_std: float=1.0, penalty_scale: float=1.0, peak_scale: float=1.0, w: ImpedanceSpectrumWeights | None=None)`** — Per-sample impedance loss (B,) and components for logging.
- **`surrogate_spectrum_loss(pred: torch.Tensor, target: torch.Tensor, *, ch0_weight: float=1.0, topk_k: int=20, topk_weight: float=7.0, under_penalty: float=2.8, freq_weight_alpha: float=2.0, dual_topk_weight: float=1.0, peak_index_weight: float=2.0, peak_mag_weight: float=1.5, num_peaks: int=8, peak_scale: float=1.0)`** — Scalar loss for occ → impedance surrogate.

## Imported by

- [[experiments.exp038_true_multi.codes.eval_val_recon]]
- [[experiments.exp038_true_multi.codes.surrogate_impedance]]
- [[experiments.exp038_true_multi.codes.train_vae_simple]]
- [[experiments.exp039_improved_heatmap.codes.codes.eval_val_recon]]
- [[experiments.exp039_improved_heatmap.codes.codes.surrogate_impedance]]
- [[experiments.exp039_improved_heatmap.codes.codes.train_vae_simple]]
- [[experiments.exp040.codes.codes.eval_val_recon]]
- [[experiments.exp040.codes.codes.surrogate_impedance]]
- [[experiments.exp040.codes.codes.train_vae_simple]]
- [[experiments.exp041.codes.codes.eval_val_recon]]
- [[experiments.exp041.codes.codes.surrogate_impedance]]
- [[experiments.exp041.codes.codes.train_vae_simple]]
- [[optimize]]

## External dependencies

`torch`
