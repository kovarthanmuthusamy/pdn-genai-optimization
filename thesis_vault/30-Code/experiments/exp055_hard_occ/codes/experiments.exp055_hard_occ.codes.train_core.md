---
title: train_core
type: code
path: experiments/exp055_hard_occ/codes/train_core.py
group: experiments/exp055_hard_occ/codes
experiment: exp055_hard_occ
loc: 1586
tags: [code, exp055_hard_occ]
---

# train_core

> exp055 training core loop (self-contained).

**Source:** `experiments/exp055_hard_occ/codes/train_core.py` · 1586 lines
**Experiment:** [[exp055_hard_occ]]

## Constants

| Name | Value |
|------|-------|
| `_ROOT` | `Path(__file__).resolve()` |
| `PROJECT_ROOT` | `next((str(p) for p in _ROOT.parents if (p / 'src_vae').is_dir() and (p / 'experiments').i…` |
| `_FRAC_FIELDS` | `(('beta_end_epoch', 'beta_end_frac'), ('beta_phase2_end_epoch', 'beta_phase2_end_frac'), …` |
| `_CURRICULUM_FRAC_FIELDS` | `(('cross_freq_start_epoch', 'cross_freq_start_frac'), ('modality_dropout_protect_heatmap_…` |
| `_ALL_SCHEDULE_FRAC_PAIRS` | `_FRAC_FIELDS + _CURRICULUM_FRAC_FIELDS` |
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
- **`_percentile_along_dim(x: torch.Tensor, q: float, dim: int=1, *, max_samples: int=2048)`** — Percentile via kthvalue; subsample long rows to cap sort cost.
- **`_downsample_maps_2x(*tensors: torch.Tensor)`** — 2× avg-pool spatial maps when H or W ≥ 128 (keeps semantics, ~4× fewer pixels).
- **`_imp_ch0(x: torch.Tensor)`**
- **`focal_bce(pred: torch.Tensor, target: torch.Tensor, gamma: float, pos_weight: torch.Tensor | None=None)`**
- **`_clip_recon_heatmap_z(recon: torch.Tensor, c: Config)`**
- **`heatmap_loss(recon: torch.Tensor, target: torch.Tensor, c: Config, ps: float, *, dynrange_weight: float | None=None, lite: bool=False)`** — Per-sample heatmap loss (B,). ``lite=True`` skips grad/lap/contrast/bg/p95 (aux paths).
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
- **`apply_curriculum_epochs(c: Config, *, skip_epoch_keys: set[str] | None=None)`** — Map ``*_frac`` → absolute ``*_epoch`` = round(num_epochs * frac).
- **`restore_curriculum_epochs_from_checkpoint(c: Config, ckpt_cfg: dict)`** — Keep phase boundaries from the run that wrote the checkpoint (resume-safe).
- **`clamp_curriculum_for_resume_epoch(c: Config, resume_epoch: int)`** — If resume is already past a phase start, do not push that start into the future.
- **`print_curriculum_summary(c: Config)`**
- **`load_experiment_config(path: Path)`** — Parse config.yaml: JSON object with optional full-line ``#`` comments.
- **`_apply_yaml_config(c: Config)`** — Merge experiment config.yaml (extension / resume knobs).
- **`_impedance_module_params(model: torch.nn.Module)`**
- **`_adamw(params, *, lr: float, c: Config, use_grad_scaler: bool=False)`** — CUDA AdamW; fused/foreach are disabled with GradScaler (fp16 amp).
- **`_build_optimizer(model: torch.nn.Module, physics: PhysicsLoss | None, c: Config, epoch_start: int, *, use_grad_scaler: bool=False)`** — AdamW with optional higher LR on impedance decoder/encoder during peak-focus phase.
- **`_append_impedance_split_csv(metrics_dir: Path, epoch: int, tr: dict, val: dict | None)`**
- **`_resolve_resume(c: Config)`**
- **`load_checkpoint(path: str, model, optimizer=None, device='cuda', physics=None)`**
- **`_ckpt_dict(epoch, model, opt, tr_loss, val, c, latent_stats, per_k, physics, sched)`**
- **`_run_epoch(model, loader, c: Config, epoch: int, beta: float, md: float, physics: PhysicsLoss | None, pw: tuple[float, float, float] | None, imp_log_std: float, hm_log_mean: float, hm_log_std: float, *, train: bool, optimizer=None, scaler=None, collect_per_k: bool=False)`**
- **`_print_interval(c: Config)`**
- **`_should_print_epoch(ep: int, c: Config, *, checkpoint: bool)`**
- **`_resolve_dataloader_workers(c: Config)`** — RAM-cached datasets: workers serialize huge tensors over IPC and starve the GPU.
- **`_should_run_validation(ep: int, num_epochs: int, checkpoint_interval: int, *, val_on_checkpoint_only: bool)`** — Full val on epoch 1, every checkpoint_interval, and final epoch.
- **`_on_stats_loaded(c: Config, raw: dict)`** — Optional hook after normalization_stats.json is applied (experiment overrides).
- **`build_vae_model(c: Config)`** — Build the training model (override in experiment-specific train_vae_simple).
- **`train_vae()`**

## Imports

- [[experiments.exp055_hard_occ.codes.dataloader_multifreq]]
- [[experiments.exp055_hard_occ.codes.distributed_train]]
- [[experiments.exp055_hard_occ.codes.eval_off_anchor]]
- [[experiments.exp055_hard_occ.codes.impedance_spectrum_loss]]
- [[experiments.exp055_hard_occ.codes.physics_loss]]
- [[experiments.exp055_hard_occ.codes.vae_poe_freq]]
- [[heatmap_z_clip]]
- [[pi_freq_utils]]
- [[vae_logger]]

## Imported by

- [[experiments.exp055_hard_occ.codes.diagnose_nan_grad]]
- [[experiments.exp055_hard_occ.codes.run_epoch_encode]]
- [[experiments.exp055_hard_occ.codes.train_vae_simple]]

## External dependencies

`src_vae`, `torch`
