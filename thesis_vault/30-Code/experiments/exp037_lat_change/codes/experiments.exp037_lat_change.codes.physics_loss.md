---
title: physics_loss
type: code
path: experiments/exp037_lat_change/codes/physics_loss.py
group: experiments/exp037_lat_change/codes
experiment: exp037_lat_change
loc: 239
tags: [code, exp037_lat_change]
---

# physics_loss

> Physics-informed loss modules for exp031 Multi-Input VAE.

**Source:** `experiments/exp037_lat_change/codes/physics_loss.py` · 239 lines
**Experiment:** [[exp037_lat_change]]

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

- **`PhysicsCritic(nn.Module)`** — Small CNN that learns the spatial effectiveness field of decap occupancy.
- **`PhysicsLoss(nn.Module)`** — Physics-informed constraint losses for the Multi-Modal VAE.

## Functions

- **`_physics_stage_weights(epoch: int, cfg: object)`** — Return per-loss weight dict for the current epoch.

## Imported by

- [[experiments.exp037_lat_change.codes.train_vae_simple]]

## External dependencies

`torch`
