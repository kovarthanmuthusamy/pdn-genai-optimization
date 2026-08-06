---
title: physics_loss
type: code
path: experiments/exp058_asymmetric_kl/codes/physics_loss.py
group: experiments/exp058_asymmetric_kl/codes
experiment: exp058_asymmetric_kl
loc: 298
tags: [code, exp058_asymmetric_kl, uncommitted]
---

# physics_loss

> Physics-informed loss modules for exp031 Multi-Input VAE.

**Source:** `experiments/exp058_asymmetric_kl/codes/physics_loss.py` · 298 lines
**Git:** uncommitted — not yet tracked
**Experiment:** [[exp058_asymmetric_kl]]

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

- [[experiments.exp058_asymmetric_kl.codes.train_core]]

## External dependencies

`src_vae`, `torch`
