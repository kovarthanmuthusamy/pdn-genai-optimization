---
title: run_epoch_encode
type: code
path: experiments/exp053_peak_log1p_losses/codes/run_epoch_encode.py
group: experiments/exp053_peak_log1p_losses/codes
experiment: exp053_peak_log1p_losses
loc: 502
tags: [code, exp053_peak_log1p_losses]
---

# run_epoch_encode

> Training epoch loop for exp053 — pixel peak blob losses + layout path.

**Source:** `experiments/exp053_peak_log1p_losses/codes/run_epoch_encode.py` · 502 lines
**Experiment:** [[exp053_peak_log1p_losses]]

## Functions

- **`_training_use_amp(c: Config, *, train: bool)`** — exp053: full fp32 train step avoids AMP backward NaNs in decoder/conditioner.
- **`_peak_phys_loss_weight(pi_norm: torch.Tensor, c: Config, base_weight: float)`** — Boost physical peak-blob loss at high MHz.
- **`_pi_norm_to_mhz(pi_norm: torch.Tensor)`** — Batch PI norm → MHz.
- **`_high_freq_mult(pi_norm: torch.Tensor, c: Config)`**
- **`_latent_distill_loss(mu_student: torch.Tensor, mu_teacher: torch.Tensor, c: Config, base, logvar_student: torch.Tensor | None=None, logvar_teacher: torch.Tensor | None=None)`** — Pull the layout (student) latent toward the encode (teacher) latent.
- **`_forward_train_batch(model, hm_enc: torch.Tensor, occ: torch.Tensor, imp: torch.Tensor, K: torch.Tensor, pi: torch.Tensor, c: Config, *, train: bool)`** — Encode for KL; decode with teacher U-Net skips on encode batches.
- **`_cross_freq_decode(c: Config, base, *, hm_enc, occ, imp, K, pi, z_decode, pi_alt, hm_alt, ps: float, force_layout_z: bool=False)`** — Encode z + teacher skips for cross-freq heatmap loss.
- **`_run_epoch(model, loader, c: Config, epoch: int, beta: float, md: float, physics: PhysicsLoss | None, pw: tuple[float, float, float] | None, imp_log_std: float, hm_log_mean: float, hm_log_std: float, *, train: bool, optimizer=None, scaler=None, collect_per_k: bool=False)`**

## Imports

- [[experiments.exp038_true_multi.codes.train_vae_simple]]
- [[experiments.exp053_peak_log1p_losses.codes.heatmap_peak_losses]]
- [[experiments.exp053_peak_log1p_losses.codes.mhz_loss_weight]]
- [[experiments.exp053_peak_log1p_losses.codes.training_guard]]
- [[pi_freq_utils]]

## Imported by

- [[experiments.exp053_peak_log1p_losses.codes.diagnose_nan_grad]]
- [[experiments.exp053_peak_log1p_losses.codes.train_vae_simple]]

## External dependencies

`src_vae`, `torch`
