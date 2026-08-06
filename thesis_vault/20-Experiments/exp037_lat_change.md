---
title: exp037_lat_change
type: experiment
status: historical
era: vae
tags: [experiment, exp037_lat_change, historical, era-vae]
---

# exp037_lat_change

**Lineage:** [[exp029_heat_private]] → **exp037_lat_change** → [[exp038_true_multi]]
**Status:** historical · **Era:** VAE era (7 Python files)

## Code modules

- [[experiments.exp037_lat_change.codes.evaluate_vae]] — evaluate_vae.py — Post-training evaluation for exp025_latent_size_change
- [[experiments.exp037_lat_change.codes.inference_vae]] — Inference script for Multi-Input VAE — exp037_lat_change.
- [[experiments.exp037_lat_change.codes.physics_loss]] — Physics-informed loss modules for exp031 Multi-Input VAE.
- [[experiments.exp037_lat_change.codes.surrogate_impedance]] — Surrogate impedance model and training script — exp037.
- [[experiments.exp037_lat_change.codes.train_vae_simple]] — Training script — Multi-Input VAE (exp037_lat_change).
- [[experiments.exp037_lat_change.codes.vae_multi_input_simple]]
- [[experiments.exp037_lat_change.codes.visualize_latent]] — visualize_latent.py — Latent space visualizations for exp025_latent_size_change

## Metrics artifacts

`experiments/exp037_lat_change/metrics/`

- `loss.csv`
- `timing.json`

## Checkpoints

`best_model.pt`, `checkpoint_epoch_100.pt`, `checkpoint_epoch_150.pt`, `checkpoint_epoch_200.pt`, `checkpoint_epoch_250.pt`, `checkpoint_epoch_300.pt`, `checkpoint_epoch_350.pt`, `checkpoint_epoch_400.pt`, `checkpoint_epoch_50.pt`, `last_model.pt`
