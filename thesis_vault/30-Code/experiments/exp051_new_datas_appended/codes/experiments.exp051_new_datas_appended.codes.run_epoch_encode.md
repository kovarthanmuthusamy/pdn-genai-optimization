---
title: run_epoch_encode
type: code
path: experiments/exp051_new_datas_appended/codes/run_epoch_encode.py
group: experiments/exp051_new_datas_appended/codes
experiment: exp051_new_datas_appended
loc: 426
tags: [code, exp051_new_datas_appended]
---

# run_epoch_encode

> Training epoch loop for exp051 — pixel peak blob losses + layout path.

**Source:** `experiments/exp051_new_datas_appended/codes/run_epoch_encode.py` · 426 lines
**Experiment:** [[exp051_new_datas_appended]]

## Functions

- **`_peak_phys_loss_weight(pi_norm: torch.Tensor, c: Config, base_weight: float)`** — Boost physical peak-blob loss at high MHz.
- **`_pi_norm_to_mhz(pi_norm: torch.Tensor)`** — Batch PI norm → MHz.
- **`_high_freq_mult(pi_norm: torch.Tensor, c: Config)`**
- **`_latent_distill_loss(mu_student: torch.Tensor, mu_teacher: torch.Tensor, c: Config, base, logvar_student: torch.Tensor | None=None, logvar_teacher: torch.Tensor | None=None)`** — Pull the layout (student) latent toward the encode (teacher) latent.
- **`_forward_train_batch(model, hm_enc: torch.Tensor, occ: torch.Tensor, imp: torch.Tensor, K: torch.Tensor, pi: torch.Tensor, c: Config, *, train: bool)`** — Encode for KL; decode with teacher U-Net skips on encode batches.
- **`_cross_freq_decode(c: Config, base, *, hm_enc, occ, imp, K, pi, z_decode, pi_alt, hm_alt, ps: float, dynrange_weight: float | None, force_layout_z: bool=False)`** — Encode z + teacher skips for cross-freq heatmap loss.
- **`_run_epoch(model, loader, c: Config, epoch: int, beta: float, md: float, physics: PhysicsLoss | None, pw: tuple[float, float, float] | None, imp_log_std: float, hm_log_mean: float, hm_log_std: float, *, train: bool, optimizer=None, scaler=None, collect_per_k: bool=False)`**

## Imports

- [[experiments.exp038_true_multi.codes.train_vae_simple]]
- [[experiments.exp051_new_datas_appended.codes.heatmap_peak_losses]]
- [[experiments.exp051_new_datas_appended.codes.spatial_metrics]]
- [[pi_freq_utils]]

## Imported by

- [[experiments.exp051_new_datas_appended.codes.train_vae_simple]]

## External dependencies

`src_vae`, `torch`
