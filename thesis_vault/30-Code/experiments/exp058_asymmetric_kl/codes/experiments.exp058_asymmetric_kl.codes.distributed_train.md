---
title: distributed_train
type: code
path: experiments/exp058_asymmetric_kl/codes/distributed_train.py
group: experiments/exp058_asymmetric_kl/codes
experiment: exp058_asymmetric_kl
loc: 123
tags: [code, exp058_asymmetric_kl, uncommitted]
---

# distributed_train

> DDP helpers for exp055 multi-GPU training.

**Source:** `experiments/exp058_asymmetric_kl/codes/distributed_train.py` · 123 lines
**Git:** uncommitted — not yet tracked
**Experiment:** [[exp058_asymmetric_kl]]

## Functions

- **`distributed_enabled()`**
- **`get_rank()`**
- **`get_world_size()`**
- **`is_main_process()`**
- **`init_distributed()`** — Return (local_rank, rank, world_size). No-op when not launched via torchrun.
- **`cleanup_distributed()`**
- **`barrier()`**
- **`unwrap_model(model: torch.nn.Module)`**
- **`wrap_ddp(model: torch.nn.Module, *, local_rank: int)`**
- **`broadcast_float(value: float, *, device: str)`**
- **`model_state_dict(model: torch.nn.Module)`**
- **`apply_ddp_config(c, *, world_size: int, local_rank: int)`** — Tune config for multi-GPU: device, LR scale, RAM cache, workers.
- **`main_process_first()`** — Run block on rank 0 first, then let other ranks proceed (barrier sync).

## Imported by

- [[experiments.exp058_asymmetric_kl.codes.run_epoch_encode]]
- [[experiments.exp058_asymmetric_kl.codes.train_core]]
- [[experiments.exp058_asymmetric_kl.codes.train_vae_simple]]

## External dependencies

`torch`
