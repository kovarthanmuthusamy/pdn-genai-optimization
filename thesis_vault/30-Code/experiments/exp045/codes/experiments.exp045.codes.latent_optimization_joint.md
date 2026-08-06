---
title: latent_optimization_joint
type: code
path: experiments/exp045/codes/latent_optimization_joint.py
group: experiments/exp045/codes
experiment: exp045
loc: 168
tags: [code, exp045]
---

# latent_optimization_joint

> Joint latent optimization: heatmap + impedance (exp045, log1p z-score norm).

**Source:** `experiments/exp045/codes/latent_optimization_joint.py` · 168 lines
**Experiment:** [[exp045]]

## Purpose

```text
Joint latent optimization: heatmap + impedance (exp045, log1p z-score norm).

Optimizes latent z so decode matches target heatmap and impedance in **train norm space**
(same as ``data_multifreq_norm``). Denorm to physical Ω only for reporting.

Run:
    .venv/bin/python experiments/exp045/codes/latent_optimization_joint.py \
        --checkpoint experiments/exp045/checkpoints/last_model.pt \
        --sample-index 0 --mhz 200 --steps 200
```

## Constants

| Name | Value |
|------|-------|
| `_ROOT` | `Path(__file__).resolve().parents[3]` |

## Functions

- **`_load_sample(data_dir: Path, index: int, device: torch.device)`**
- **`optimize_latent(model, *, hm_tgt: torch.Tensor, occ: torch.Tensor, imp_tgt: torch.Tensor, K: torch.Tensor, pi: torch.Tensor, steps: int, lr: float, w_hm: float, w_imp: float, huber_delta: float)`**
- **`main()`**

## Imports

- [[dataloader]]
- [[exp045_eval_common]]
- [[pi_freq_utils]]

## External dependencies

`src_vae`, `torch`
