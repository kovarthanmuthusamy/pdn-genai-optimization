---
title: exp039_eval_common
type: code
path: experiments/exp040/codes/exp039_eval_common.py
group: experiments/exp040/codes
experiment: exp040
loc: 185
tags: [code, exp040]
---

# exp039_eval_common

> Shared paths, model loading, and PI_freq helpers for exp039 evaluation scripts.

**Source:** `experiments/exp040/codes/exp039_eval_common.py` · 185 lines
**Experiment:** [[exp040]]

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
- **`pi_norm_to_mhz(pi_norm: np.ndarray | float)`** — Invert log10-normalised PI_freq tensor → MHz.
- **`load_model(checkpoint_path: str | Path, device: torch.device)`**
- **`encode_dataset(model: MultiInputVAE, data_dir: str | Path, device: torch.device, *, max_samples: int=30000, batch_size: int=128, num_workers: int=4, seed: int=42, collect_experts: bool=True)`**
- **`nearest_anchor_index(mhz: float)`**
- **`pi_tensor_mhz(mhz: float, batch: int, device: torch.device)`**

## Imports

- [[dataloader]]
- [[experiments.exp038_true_multi.codes.vae_multi_input_simple]]
- [[multifreq_anchors]]
- [[pi_freq_utils]]

## External dependencies

`numpy`, `src_vae`, `torch`
