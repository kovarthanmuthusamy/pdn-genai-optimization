---
title: exp045_eval_common
type: code
path: experiments/exp045/codes/exp045_eval_common.py
group: experiments/exp045/codes
experiment: exp045
loc: 188
tags: [code, exp045]
---

# exp045_eval_common

> Shared paths, model loading, and PI_freq helpers for exp045 evaluation scripts.

**Source:** `experiments/exp045/codes/exp045_eval_common.py` · 188 lines
**Experiment:** [[exp045]]

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
- [[experiments.exp045.codes.vae_poe_freq]]
- [[multifreq_anchors]]
- [[pi_freq_utils]]

## Imported by

- [[experiments.exp045.codes.latent_optimization_joint]]

## External dependencies

`numpy`, `src_vae`, `torch`
