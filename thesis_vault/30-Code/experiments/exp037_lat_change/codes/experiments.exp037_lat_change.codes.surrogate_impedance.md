---
title: surrogate_impedance
type: code
path: experiments/exp037_lat_change/codes/surrogate_impedance.py
group: experiments/exp037_lat_change/codes
experiment: exp037_lat_change
loc: 328
tags: [code, exp037_lat_change]
---

# surrogate_impedance

> Surrogate impedance model and training script — exp037.

**Source:** `experiments/exp037_lat_change/codes/surrogate_impedance.py` · 328 lines
**Experiment:** [[exp037_lat_change]]

## Purpose

```text
Surrogate impedance model and training script — exp037.

Direct mapping:  occ (52,) → impedance (1, 231) log-z spectrum
No VAE bottleneck → can learn exact resonance structure.

The surrogate is used during latent optimization:
    z → VAE occ decoder → occ_binary (STE) → surrogate → accurate impedance → score

Run:
    python experiments/exp037_lat_change/codes/surrogate_impedance.py
```

## Constants

| Name | Value |
|------|-------|
| `PROJECT_ROOT` | `next((str(p) for p in _here.parents if (p / 'datasets').is_dir() and (p / 'experiments').…` |

## Classes

- **`SurrConfig`**
- **`OccImpDataset(Dataset)`** — Load occupancy → impedance (ch0) pairs. PI_freq in files is ignored (heatmap-only cond).
- **`SurrogateImpedanceNet(nn.Module)`** — Direct occ → impedance (1, 231). K/decap layout only — not PI_freq.

## Functions

- **`_get_indices(data_dir: str)`**
- **`surrogate_loss(pred: torch.Tensor, target: torch.Tensor, cfg: SurrConfig)`** — Per-batch loss for (B, 1, 231) predictions.
- **`train_surrogate(cfg: SurrConfig | None=None)`**
- **`load_surrogate(ckpt_path: str | Path, device: str | torch.device='cuda')`** — Load a trained surrogate model from checkpoint.

## External dependencies

`numpy`, `torch`
