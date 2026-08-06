---
title: diagnose_exp044_sweep
type: code
path: scratch/diagnose_exp044_sweep.py
group: scratch
loc: 94
tags: [code, scratch]
---

# diagnose_exp044_sweep

> Quick encode vs layout diagnostic for exp044 at off-anchor MHz.

**Source:** `scratch/diagnose_exp044_sweep.py` · 94 lines

## Constants

| Name | Value |
|------|-------|
| `CKPT` | `_ROOT / 'experiments/exp044/checkpoints/last_model.pt'` |
| `TARGET_MHZ` | `330.0` |
| `K_FILTER` | `30` |
| `MAX_ROWS` | `32` |

## Functions

- **`fg_mse(recon: torch.Tensor, target: torch.Tensor, bg: float)`**
- **`main()`**

## Imports

- [[experiments.exp038_true_multi.codes.dataloader_multifreq]]
- [[experiments.exp044.codes.inference_vae]]
- [[pi_freq_utils]]
- [[repo_paths]]

## External dependencies

`experiments`, `repo_paths`, `src_vae`, `statistics`, `torch`
