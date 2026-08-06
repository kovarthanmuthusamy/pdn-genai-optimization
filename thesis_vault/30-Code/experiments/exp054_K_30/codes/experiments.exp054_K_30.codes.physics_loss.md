---
title: physics_loss
type: code
path: experiments/exp054_K_30/codes/physics_loss.py
group: experiments/exp054_K_30/codes
experiment: exp054_K_30
loc: 298
tags: [code, exp054_K_30]
---

# physics_loss

> Physics-informed loss modules for exp031 Multi-Input VAE.

**Source:** `experiments/exp054_K_30/codes/physics_loss.py` · 298 lines
**Experiment:** [[exp054_K_30]]

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

- [[experiments.exp054_K_30.codes.train_core]]

## External dependencies

`src_vae`, `torch`
