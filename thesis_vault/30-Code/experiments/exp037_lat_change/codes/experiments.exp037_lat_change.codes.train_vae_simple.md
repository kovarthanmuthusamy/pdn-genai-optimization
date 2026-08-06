---
title: train_vae_simple
type: code
path: experiments/exp037_lat_change/codes/train_vae_simple.py
group: experiments/exp037_lat_change/codes
experiment: exp037_lat_change
loc: 1096
tags: [code, exp037_lat_change]
---

# train_vae_simple

> Training script — Multi-Input VAE (exp037_lat_change).

**Source:** `experiments/exp037_lat_change/codes/train_vae_simple.py` · 1096 lines
**Experiment:** [[exp037_lat_change]]

## Constants

| Name | Value |
|------|-------|
| `_LAP_KERNEL` | `torch.tensor([[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]], dtype=torch.float3…` |

## Classes

- **`Config`**

## Functions

- **`compute_beta(epoch: int, cfg: Config)`**
- **`compute_modality_dropout(epoch: int, cfg: Config)`**
- **`compute_focal_gamma(epoch: int, cfg: Config)`** — Linear warmup of focal gamma: 0→cfg.occupancy_focal_gamma.
- **`_physics_stage_weights(epoch: int, cfg: Config)`**
- **`focal_bce_per_sample(pred: torch.Tensor, target: torch.Tensor, gamma: float, pos_weight: 'torch.Tensor | None'=None)`** — Focal BCE per sample (B,). pred = raw logits.
- **`_lap_kernel(device, dtype=torch.float32)`** — Return _LAP_KERNEL on the requested device, caching after first use.
- **`_penalty_scale(epoch: int, cfg)`** — Linear ramp 0→1 for noisy penalty terms (topk, concavity, Laplacian,
- **`heatmap_loss_per_sample(recon: torch.Tensor, target: torch.Tensor, cfg: Config, penalty_scale: float=1.0)`** — Foreground Huber + gradient + Laplacian sharpness + fg/bg contrast. Returns (B,).
- **`_imp_ch0(x: torch.Tensor)`** — Log-z impedance channel: (B, 1, 231) or (B, 231) → (B, 231).
- **`impedance_loss_per_sample(recon: torch.Tensor, target: torch.Tensor, cfg: Config, *, delta_raw: float=1.0, delta_deriv: float=1.0, imp_log_std: float=1.0, penalty_scale: float=1.0)`** — Multi-peak-aware impedance loss.  Returns (B,).
- **`_weighted_mean(v: torch.Tensor, w: torch.Tensor)`**
- **`_k_loss_weights(K: torch.Tensor, cfg: Config)`**
- **`_expert_kl_loss(expert_stats: dict)`**
- **`vae_loss(recon_hm, recon_occ, recon_imp, target_hm, target_occ, target_imp, mu, logvar, beta: float, cfg: Config, expert_stats=None, *, epoch: int=0, K: torch.Tensor | None=None, apply_k_weights: bool=True, physics: 'PhysicsLoss | None'=None, physics_weights: tuple[float, ...] | None=None, imp_log_std: float=1.0, penalty_scale: float=1.0)`**
- **`_init_trackers(modalities=('heatmap', 'occupancy', 'impedance'))`**
- **`_update_scalar(s: dict, t: torch.Tensor)`**
- **`_finalize_scalar(s: dict, n: int)`**
- **`_update_modality_stats(per_mod: dict, expert_stats: dict)`**
- **`_finalize_modality_stats(per_mod: dict, n: int)`**
- **`_device_type(device: str)`**
- **`_autocast_dtype(cfg: Config)`**
- **`_prepare_batch(batch: dict, cfg: Config)`**
- **`load_checkpoint(path: str, model, optimizer=None, device='cuda', physics=None)`**
- **`_build_checkpoint(epoch, model, optimizer, train_loss, val, cfg, latent_stats, per_K, physics=None, scheduler=None)`**
- **`_cross_modal_loss(model, hm_enc, occ, imp, hm, K, cfg: Config, imp_log_std: float=1.0, penalty_scale: float=1.0)`** — Cross-modal reconstruction loss (heatmap + impedance sources), averaged.
- **`train_epoch(model, loader, optimizer, cfg: Config, epoch: int, beta: float, modality_dropout: float, physics=None, physics_weights=None, imp_log_std=1.0, scaler=None)`**
- **`validate(model, loader, cfg: Config, beta: float, epoch: int, physics=None, physics_weights=None, imp_log_std=1.0)`**
- **`train_vae()`**

## Imports

- [[dataloader]]
- [[experiments.exp037_lat_change.codes.physics_loss]]
- [[experiments.exp037_lat_change.codes.vae_multi_input_simple]]
- [[vae_logger]]

## External dependencies

`numpy`, `src_vae`, `torch`
