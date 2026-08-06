---
title: inspect_sweep_artifacts
type: code
path: experiments/exp040/codes/inspect_sweep_artifacts.py
group: experiments/exp040/codes
experiment: exp040
loc: 195
tags: [code, exp040]
---

# inspect_sweep_artifacts

> Diagnose exp040 sweep artifacts: fixed hotspot, K/MHz invariance, residual vs factorized.

**Source:** `experiments/exp040/codes/inspect_sweep_artifacts.py` · 195 lines
**Experiment:** [[exp040]]

## Purpose

```text
Diagnose exp040 sweep artifacts: fixed hotspot, K/MHz invariance, residual vs factorized.

Run from repo root (with project venv + CUDA optional):
    python experiments/exp040/codes/inspect_sweep_artifacts.py
    python experiments/exp040/codes/inspect_sweep_artifacts.py --ckpt experiments/exp040/checkpoints/checkpoint_epoch_250.pt
```

## Constants

| Name | Value |
|------|-------|
| `_ROOT` | `Path(__file__).resolve().parents[3]` |

## Functions

- **`_hm_phys(hm_z: torch.Tensor, engine)`**
- **`_peak_yx(plane: np.ndarray, mask: np.ndarray)`**
- **`main()`**

## Imports

- [[experiments.exp040.codes.inference_vae]]
- [[heatmap_z_clip]]
- [[pi_freq_utils]]

## External dependencies

`numpy`, `src_vae`, `torch`
