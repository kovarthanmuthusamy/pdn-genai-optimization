---
title: exp043_eval_common
type: code
path: experiments/exp043/codes/exp043_eval_common.py
group: experiments/exp043/codes
experiment: exp043
loc: 197
tags: [code, exp043]
---

# exp043_eval_common

> Shared model loading, paths, and PI_freq helpers for exp043 eval scripts.

**Source:** `experiments/exp043/codes/exp043_eval_common.py` · 197 lines
**Experiment:** [[exp043]]

## Purpose

```text
Shared model loading, paths, and PI_freq helpers for exp043 eval scripts.

Run:
    Import only — ``from experiments.exp043.codes.exp043_eval_common import load_model``.
```

## Constants

| Name | Value |
|------|-------|
| `_ROOT` | `Path(__file__).resolve().parents[3]` |
| `EXP_DIR` | `Path(__file__).resolve().parents[1]` |
| `PROJECT_ROOT` | `_ROOT` |
| `DEFAULT_CKPT` | `EXP_DIR / 'checkpoints/last_model.pt'` |

## Functions

- **`load_exp_config()`**
- **`resolve_paths(cfg: dict[str, Any] | None=None)`**
- **`pi_norm_to_mhz(pi_norm: np.ndarray | float)`**
- **`load_model(checkpoint_path: str | Path, device: torch.device)`**
- **`encode_dataset(model: MultiInputVAEPoeFreq, data_dir: str | Path, device: torch.device, *, max_samples: int=30000, batch_size: int=128, num_workers: int=4, seed: int=42, collect_experts: bool=True)`**
- **`nearest_anchor_index(mhz: float)`**
- **`pi_tensor_mhz(mhz: float, batch: int, device: torch.device)`**
- **`heatmap_to_physical(hm_norm: torch.Tensor, norm_stats: dict[str, Any])`** — Denorm heatmap batch to Ω (z-score or global-max).

## Imports

- [[dataloader]]
- [[experiments.exp043.codes.vae_poe_freq]]
- [[heatmap_z_clip]]
- [[multifreq_anchors]]
- [[pi_freq_utils]]

## External dependencies

`numpy`, `src_vae`, `torch`
