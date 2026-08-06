---
title: train_vae_simple
type: code
path: experiments/exp041/codes/codes/train_vae_simple.py
group: experiments/exp041/codes/codes
experiment: exp041
loc: 1287
tags: [code, exp041]
---

# train_vae_simple

> Training — Multi-Input VAE (exp038_true_multi): multifreq PI heatmaps + peak losses.

**Source:** `experiments/exp041/codes/codes/train_vae_simple.py` · 1287 lines
**Experiment:** [[exp041]]

## Constants

| Name | Value |
|------|-------|
| `_ROOT` | `Path(__file__).resolve()` |
| `PROJECT_ROOT` | `next((str(p) for p in _ROOT.parents if (p / 'src_vae').is_dir() and (p / 'experiments').i…` |
| `_FRAC_FIELDS` | `(('beta_end_epoch', 'beta_end_frac'), ('beta_phase2_end_epoch', 'beta_phase2_end_frac'), …` |
| `_CURRICULUM_FRAC_FIELDS` | `(('cross_freq_start_epoch', 'cross_freq_start_frac'), ('modality_dropout_protect_heatmap_…` |
| `LOSS_KEYS` | `('total_loss', 'recon_loss', 'kl_loss', 'kl_gaussian', 'heatmap_loss', 'occupancy_loss', …` |
| `_LAP` | `torch.tensor([[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]], dtype=torch.float3…` |

## Classes

- **`Config`**

## Functions

- **`_lerp(a: float, b: float, t: float)`**
- **`compute_beta(epoch: int, c: Config)`**
- **`compute_modality_dropout(epoch: int, c: Config)`**
- **`_phase_weights(epoch: int, c: Config)`** — Per-epoch loss weights; heatmap_focus phase boosts heatmap / cross-freq.
- **`_jitter_pi_freq_norm(pi: torch.Tensor, c: Config, *, train: bool)`** — Log10-jitter in norm space; decode-only (target heatmap unchanged).
- **`compute_focal_gamma(epoch: int, c: Config)`**
- **`_physics_weights(epoch: int, c: Config)`**
- **`_penalty_scale(epoch: int, c: Config)`**
- **`_lap_k(device, dtype)`**
- **`_imp_ch0(x: torch.Tensor)`**
- **`focal_bce(pred: torch.Tensor, target: torch.Tensor, gamma: float, pos_weight: torch.Tensor | None=None)`**
- **`heatmap_loss(recon: torch.Tensor, target: torch.Tensor, c: Config, ps: float, *, dynrange_weight: float | None=None)`**
- **`heatmap_phys_amplitude_loss(recon: torch.Tensor, target: torch.Tensor, hm_log_mean: float, hm_log_std: float, c: Config)`** — Foreground p99 in physical Ω — penalize under-prediction of peak level.
- **`_impedance_weights(c: Config)`**
- **`_peak_loss_scale(epoch: int, c: Config)`** — Ramp peak/dual-topk terms in after resume (0 before start_epoch).
- **`impedance_loss(recon: torch.Tensor, target: torch.Tensor, c: Config, imp_log_std: float, ps: float, *, epoch: int)`**
- **`_k_weights(K: torch.Tensor, c: Config)`**
- **`_wmean(v: torch.Tensor, w: torch.Tensor)`**
- **`vae_loss(recon_hm, recon_occ, recon_imp, tgt_hm, tgt_occ, tgt_imp, mu, logvar, beta: float, c: Config, expert_stats, *, epoch: int, K: torch.Tensor | None, physics: PhysicsLoss | None, pw: tuple[float, float, float] | None, pi_freq: torch.Tensor | None, imp_log_std: float, ps: float, apply_k: bool, weight_overrides: dict[str, float] | None=None)`**
- **`_gpu_supports_native_bf16(device_index: int=0)`**
- **`_adapt_amp_for_gpu(c: Config)`** — Match amp/tf32 to GPU so torch.compile + autocast avoid Volta bf16 warnings.
- **`_amp_dtype(c: Config)`**
- **`_cross_freq_heatmap_loss(model, z: torch.Tensor, K: torch.Tensor, pi_alt: torch.Tensor, hm_alt: torch.Tensor, c: Config, ps: float, *, dynrange_weight: float | None=None)`** — Decode same z at alternate PI_freq; target is ground-truth heatmap at that MHz.
- **`_forward_train_batch(model, hm_enc: torch.Tensor, occ: torch.Tensor, imp: torch.Tensor, K: torch.Tensor, pi: torch.Tensor, c: Config, *, train: bool)`** — Encode for KL; decode from layout or posterior z with optional PI_freq jitter.
- **`_prepare_batch(batch: dict, c: Config)`**
- **`_cross_modal(model, hm_enc, occ, imp, hm, K, pi, c: Config, imp_std: float, ps: float)`**
- **`_clean_previous_run(exp: Path)`** — Remove prior training artifacts when starting fresh (not resuming).
- **`_apply_yaml_config(c: Config)`** — Merge experiment config.yaml (extension / resume knobs).
- **`_impedance_module_params(model: torch.nn.Module)`**
- **`_build_optimizer(model: torch.nn.Module, physics: PhysicsLoss | None, c: Config, epoch_start: int)`** — AdamW with optional higher LR on impedance decoder/encoder during peak-focus phase.
- **`_append_impedance_split_csv(metrics_dir: Path, epoch: int, tr: dict, val: dict | None)`**
- **`_resolve_resume(c: Config)`**
- **`load_checkpoint(path: str, model, optimizer=None, device='cuda', physics=None)`**
- **`_ckpt_dict(epoch, model, opt, tr_loss, val, c, latent_stats, per_k, physics, sched)`**
- **`_run_epoch(model, loader, c: Config, epoch: int, beta: float, md: float, physics: PhysicsLoss | None, pw: tuple[float, float, float] | None, imp_log_std: float, hm_log_mean: float, hm_log_std: float, *, train: bool, optimizer=None, scaler=None, collect_per_k: bool=False)`**
- **`_should_run_validation(ep: int, num_epochs: int, checkpoint_interval: int, *, val_on_checkpoint_only: bool)`** — Full val on epoch 1, every checkpoint_interval, and final epoch.
- **`train_vae()`**

## Imports

- [[experiments.exp038_true_multi.codes.dataloader_multifreq]]
- [[experiments.exp038_true_multi.codes.eval_cross_freq]]
- [[experiments.exp038_true_multi.codes.impedance_spectrum_loss]]
- [[experiments.exp038_true_multi.codes.physics_loss]]
- [[experiments.exp038_true_multi.codes.vae_multi_input_simple]]
- [[pi_freq_utils]]
- [[vae_logger]]

## External dependencies

`src_vae`, `torch`
