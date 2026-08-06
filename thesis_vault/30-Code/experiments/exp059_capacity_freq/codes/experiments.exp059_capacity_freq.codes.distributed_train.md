---
title: distributed_train
type: code
path: experiments/exp059_capacity_freq/codes/distributed_train.py
group: experiments/exp059_capacity_freq/codes
experiment: exp059_capacity_freq
loc: 129
tags: [code, exp059_capacity_freq, uncommitted]
---

# distributed_train

> DDP helpers for exp055 multi-GPU training.

**Source:** `experiments/exp059_capacity_freq/codes/distributed_train.py` · 129 lines
**Git:** uncommitted — not yet tracked
**Experiment:** [[exp059_capacity_freq]]

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

- [[experiments.exp059_capacity_freq.codes.eval_spatial_metrics]]
- [[experiments.exp059_capacity_freq.codes.run_epoch_encode]]
- [[experiments.exp059_capacity_freq.codes.train_core]]
- [[experiments.exp059_capacity_freq.codes.train_vae_simple]]

## External dependencies

`torch`
