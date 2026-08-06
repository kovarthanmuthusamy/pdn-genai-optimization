---
title: physics_loss
type: code
path: experiments/exp038_true_multi/codes/physics_loss.py
group: experiments/exp038_true_multi/codes
experiment: exp038_true_multi
loc: 298
tags: [code, exp038_true_multi]
---

# physics_loss

> Physics-informed loss modules for exp031 Multi-Input VAE.

**Source:** `experiments/exp038_true_multi/codes/physics_loss.py` · 298 lines
**Experiment:** [[exp038_true_multi]]

## Purpose

```text
Physics-informed loss modules for exp031 Multi-Input VAE.

Exported symbols:
    PhysicsCritic      — occupancy grid → spatial effectiveness map (CNN)
    PhysicsLoss        — combined module: all critics + all loss methods
    _physics_stage_weights — staged training weight schedule

These are imported by train_vae_simple.py and by Latent_opm/latent_optimization_impedance.py.
```

## Classes

- **`PhysicsCritic(nn.Module)`** — Dual-branch CNN: decap occupancy grid + PI_freq → spatial effectiveness map.
- **`PhysicsLoss(nn.Module)`** — Physics-informed constraint losses for the Multi-Modal VAE (exp035).

## Functions

- **`_physics_stage_weights(epoch: int, cfg: object)`** — Return per-loss weight dict for the current epoch.

## Imports

- [[pi_freq_utils]]

## Imported by

- [[experiments.exp038_true_multi.codes.save_epoch1_losses]]
- [[experiments.exp038_true_multi.codes.train_vae_simple]]
- [[experiments.exp039_improved_heatmap.codes.codes.save_epoch1_losses]]
- [[experiments.exp039_improved_heatmap.codes.codes.train_vae_simple]]
- [[experiments.exp039_improved_heatmap.codes.save_epoch1_losses]]
- [[experiments.exp040.codes.codes.save_epoch1_losses]]
- [[experiments.exp040.codes.codes.train_vae_simple]]
- [[experiments.exp040.codes.save_epoch1_losses]]
- [[experiments.exp041.codes.codes.save_epoch1_losses]]
- [[experiments.exp041.codes.codes.train_vae_simple]]
- [[experiments.exp041.codes.save_epoch1_losses]]
- [[experiments.exp052_unbounded_pearson.codes.diagnose_nan_grad]]
- [[experiments.exp053_peak_log1p_losses.codes.diagnose_nan_grad]]

## External dependencies

`src_vae`, `torch`
