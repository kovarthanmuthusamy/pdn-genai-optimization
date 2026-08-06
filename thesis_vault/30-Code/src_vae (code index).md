---
title: src_vae (code index)
type: index
tags: [index, code, src_vae]
---

# src_vae/ — code index

16 modules.

## `src_vae/`

- [[src_vae.__init__]]

## `src_vae/loss/`

- [[vae_loss]] — Active multi-input VAE loss with uncertainty weighting.
- [[vae_loss_backup]] — Extended multi-input VAE loss (backup / reference implementation).

## `src_vae/model/`

- [[vae_multi_input]] — Multi-input VAE with mid-layer fusion and hierarchical decoder.
- [[src_vae.model.vae_multi_input_simple]] — Simplified multi-input VAE (heatmap + occupancy + impedance).

## `src_vae/others/`

- [[src_vae.others.__init__]]
- [[dataloader]] — PyTorch dataset and data loaders for multi-input VAE training.
- [[heatmap_gmax_norm]] — Global-max heatmap normalization utilities.
- [[heatmap_z_clip]] — Log-z heatmap clipping and denormalization helpers.
- [[llm_agent]] — DeepSeek LLM agent for autonomous training analysis and config fixes.
- [[model_to_config]] *(runnable)* — Convert PyTorch model checkpoints to YAML experiment configs.
- [[multifreq_anchors]] — Shared PI frequency anchor MHz list and label helpers.
- [[multifreq_layout_store]] — Layout-centric on-disk layout for multifreq PI datasets.
- [[norm_stats]] — Centralized normalization / denormalization from ``normalization_stats.json``.
- [[pi_freq_utils]] — PI frequency conditioning: MHz/Hz → model scalar in [0, 1].
- [[vae_logger]] — VAE training metrics logger (CSV, plots, checkpoints, console).
