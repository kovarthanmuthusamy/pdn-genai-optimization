---
title: surrogate_impedance
type: code
path: experiments/exp038_true_multi/codes/surrogate_impedance.py
group: experiments/exp038_true_multi/codes
experiment: exp038_true_multi
loc: 278
tags: [code, exp038_true_multi]
---

# surrogate_impedance

> Surrogate impedance model and training script — exp038_true_multi.

**Source:** `experiments/exp038_true_multi/codes/surrogate_impedance.py` · 278 lines
**Experiment:** [[exp038_true_multi]]

## Purpose

```text
Surrogate impedance model and training script — exp038_true_multi.

Direct mapping:  occ (52,) → impedance (1, 231) log-z spectrum
PI_freq is not used (heatmap-only conditioning in the VAE).

Run:
    python experiments/exp038_true_multi/codes/surrogate_impedance.py
```

## Constants

| Name | Value |
|------|-------|
| `PROJECT_ROOT` | `next((str(p) for p in _here.parents if (p / 'datasets').is_dir() and (p / 'experiments').…` |

## Classes

- **`SurrConfig`**
- **`OccImpDataset(Dataset)`** — occ → impedance (ch0). PI_freq in files is ignored.
- **`SurrogateImpedanceNet(nn.Module)`** — Direct occ → impedance (1, 231). K/decap layout only — not PI_freq.

## Functions

- **`_get_indices(data_dir: str)`**
- **`surrogate_loss(pred: torch.Tensor, target: torch.Tensor, cfg: SurrConfig)`**
- **`train_surrogate(cfg: SurrConfig | None=None)`**
- **`load_surrogate(ckpt_path: str | Path, device: str | torch.device='cuda')`**

## Imports

- [[experiments.exp038_true_multi.codes.impedance_spectrum_loss]]

## Imported by

- [[optimize]]

## External dependencies

`numpy`, `torch`
