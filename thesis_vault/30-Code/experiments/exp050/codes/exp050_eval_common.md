---
title: exp050_eval_common
type: code
path: experiments/exp050/codes/exp050_eval_common.py
group: experiments/exp050/codes
experiment: exp050
loc: 191
tags: [code, exp050]
---

# exp050_eval_common

> Shared paths, model loading, and PI_freq helpers for exp050 evaluation scripts.

**Source:** `experiments/exp050/codes/exp050_eval_common.py` · 191 lines
**Experiment:** [[exp050]]

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

## Imports

- [[dataloader]]
- [[experiments.exp050.codes.vae_poe_freq]]
- [[multifreq_anchors]]
- [[pi_freq_utils]]

## Imported by

- [[experiments.exp050.codes.eval_real_data_sweep]]
- [[experiments.exp050.codes.eval_train_vs_val]]

## External dependencies

`numpy`, `src_vae`, `torch`
