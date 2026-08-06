---
title: impedance_spectrum_loss
type: code
path: experiments/exp055_hard_occ/codes/impedance_spectrum_loss.py
group: experiments/exp055_hard_occ/codes
experiment: exp055_hard_occ
loc: 120
tags: [code, exp055_hard_occ]
---

# impedance_spectrum_loss

> Impedance spectrum loss for exp055 — lean stack, no redundant terms.

**Source:** `experiments/exp055_hard_occ/codes/impedance_spectrum_loss.py` · 120 lines
**Experiment:** [[exp055_hard_occ]]

## Purpose

```text
Impedance spectrum loss for exp055 — lean stack, no redundant terms.

Terms:
  raw      — Huber on full spectrum, resonance-upweighted (freq_weight_alpha)
  deriv    — first-derivative (slope) match
  topk     — target top-k bin MSE with under-prediction penalty
  peak_idx — local peak index alignment (ramped via peak_scale)
```

## Classes

- **`ImpedanceSpectrumWeights`**

## Functions

- **`imp_ch0(x: torch.Tensor)`**
- **`resonance_freq_weights(target: torch.Tensor, alpha: float)`**
- **`local_peak_indices(x: torch.Tensor, num_peaks: int)`**
- **`_greedy_peak_index_loss(ti_n: torch.Tensor, ri_n: torch.Tensor, *, delta: float=0.08)`**
- **`peak_index_loss(recon: torch.Tensor, target: torch.Tensor, num_peaks: int)`**
- **`impedance_spectrum_loss(recon: torch.Tensor, target: torch.Tensor, *, imp_log_std: float=1.0, penalty_scale: float=1.0, peak_scale: float=1.0, w: ImpedanceSpectrumWeights | None=None)`**

## Imported by

- [[experiments.exp055_hard_occ.codes.train_core]]
- [[experiments.exp055_hard_occ.codes.train_vae_simple]]

## External dependencies

`torch`
