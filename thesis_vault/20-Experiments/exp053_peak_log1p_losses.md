---
title: exp053_peak_log1p_losses
type: experiment
status: historical
era: vae
tags: [experiment, exp053_peak_log1p_losses, historical, era-vae]
---

# exp053_peak_log1p_losses

**Lineage:** [[exp052_unbounded_pearson]] → **exp053_peak_log1p_losses** → [[exp054_K_30]]
**Status:** historical · **Era:** VAE era (19 Python files)

## Key hyperparameters

| Key | Value |
|-----|-------|
| `latent_dim` | `65` |
| `heatmap_private_dim` | `15` |
| `cond_dim` | `8` |
| `freq_fourier_features` | `8` |
| `num_epochs` | `600` |
| `batch_size` | `160` |
| `learning_rate` | `3e-05` |
| `train_split` | `0.9` |
| `split_by_design` | `True` |
| `layout_train_prob` | `0.55` |
| `heatmap_weight` | `5.0` |
| `impedance_weight` | `1.5` |
| `occupancy_weight` | `7.0` |
| `cross_freq_weight` | `1.0` |
| `data_dir` | `datasets/data_multifreq_train_norm_unbounded` |
| `resume_checkpoint` | `experiments/exp053_peak_log1p_losses/checkpoints/checkpoint_epoch_450.pt` |

*Full config: `experiments/exp053_peak_log1p_losses/config.yaml` (212 keys)*

## Notes (from `experiments/exp053_peak_log1p_losses/notes.md`)

# exp053 — resume ep325 with boosted peak losses

**Mode:** resume from `checkpoint_epoch_325.pt`, run ep326→450 with stronger log1p peak stack.

```bash
cd /home/ubuntu/genai_pdn
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m experiments.exp053_peak_log1p_losses.codes.train_vae_simple
```

**Peak tweaks (vs ep1–325):**
- `heatmap_peak_log1p_weight`: 2.75 (was 1.5)
- `heatmap_peak_hotspot_weight`: 2.5 (was 2.0)
- `heatmap_peak_max_log1p_weight`: 1.25 (was 1.0)
- `heatmap_peak_centroid_weight`: 1.875 (was 1.5)
- `heatmap_mhz_loss_weight_max`: 3.0 (was 2.5)

**Metrics:** `metrics/heatmap_peak_split.csv` logs tier_a / peak_log1p / phys_blob per checkpoint.

See `docs/EXP053_FRESH_TRAINING.md`, `docs/EXP053_TRAINING_DIAG_EP350.md`.

## Code modules

- [[experiments.exp053_peak_log1p_losses.__init__]]
- [[experiments.exp053_peak_log1p_losses.codes.__init__]]
- [[experiments.exp053_peak_log1p_losses.codes.dataloader_multifreq]] — exp052 multifreq dataloader — high-MHz cross-freq bias + append-tag curriculum sampling.
- [[experiments.exp053_peak_log1p_losses.codes.diagnose_nan_grad]] — Isolate which loss term / layer produces non-finite gradients (exp052).
- [[experiments.exp053_peak_log1p_losses.codes.eval_real_data_sweep]] — Evaluate exp046 heatmap quality using REAL dataset layouts.
- [[experiments.exp053_peak_log1p_losses.codes.eval_spatial_metrics]] — Off-anchor eval with spatial metrics (Pearson r, peak location) + early-stop hook.
- [[experiments.exp053_peak_log1p_losses.codes.eval_train_vs_val]] — Diagnostic: compare layout_cross vs encode_cross on TRAIN vs VAL designs.
- [[exp053_eval_common]] — Shared paths, model loading, and PI_freq helpers for exp053 evaluation scripts.
- [[experiments.exp053_peak_log1p_losses.codes.heatmap_peak_losses]] — exp053 heatmap losses: Pearson + grad + log1p peak stack (hotspot / max / centroid / blob).
- [[experiments.exp053_peak_log1p_losses.codes.inference_vae]] — Inference script for Multi-Input VAE — exp052.
- [[experiments.exp053_peak_log1p_losses.codes.mhz_loss_weight]] — Per-sample heatmap loss multipliers vs PI frequency (MHz).
- [[experiments.exp053_peak_log1p_losses.codes.run_epoch_encode]] — Training epoch loop for exp053 — pixel peak blob losses + layout path.
- [[experiments.exp053_peak_log1p_losses.codes.sampler_curriculum]] — Per-epoch train sampling and layout-path curriculum for exp053.
- [[experiments.exp053_peak_log1p_losses.codes.spatial_metrics]] — Foreground spatial metrics — Pearson correlation and soft peak location.
- [[experiments.exp053_peak_log1p_losses.codes.synthetic_freq_blend]] — Synthetic between-anchor heatmap targets for multifreq training (exp045).
- [[experiments.exp053_peak_log1p_losses.codes.train_vae_simple]] — Train VAE for exp053 — Pearson + grad + log1p peak stack + target-only phys blob + peak boost (resume 450→600 training with log1p peak stack).
- [[experiments.exp053_peak_log1p_losses.codes.training_guard]] — Finite-loss guards for exp053 training (NaN / Inf prevention).
- [[experiments.exp053_peak_log1p_losses.codes.vae_multi_input_simple]]
- [[experiments.exp053_peak_log1p_losses.codes.vae_poe_freq]] — Multi-input VAE — exp045: U-Net skips + freq PoE private dims + encode-first.

## Metrics artifacts

`experiments/exp053_peak_log1p_losses/metrics/`

- `epoch_timing.csv`
- `heatmap_peak_split.csv`
- `impedance_split.csv`
- `loss.csv`
- `off_anchor_eval_epoch_1.csv`
- `off_anchor_eval_epoch_100.csv`
- `off_anchor_eval_epoch_125.csv`
- `off_anchor_eval_epoch_150.csv`
- `off_anchor_eval_epoch_175.csv`
- `off_anchor_eval_epoch_200.csv`
- `off_anchor_eval_epoch_225.csv`
- `off_anchor_eval_epoch_25.csv`
- `off_anchor_eval_epoch_250.csv`
- `off_anchor_eval_epoch_275.csv`
- `off_anchor_eval_epoch_300.csv`

## Checkpoints

`checkpoint_epoch_1.pt`, `checkpoint_epoch_100.pt`, `checkpoint_epoch_125.pt`, `checkpoint_epoch_150.pt`, `checkpoint_epoch_175.pt`, `checkpoint_epoch_200.pt`, `checkpoint_epoch_225.pt`, `checkpoint_epoch_25.pt`, `checkpoint_epoch_250.pt`, `checkpoint_epoch_275.pt`, `checkpoint_epoch_300.pt`, `checkpoint_epoch_325.pt`, `checkpoint_epoch_350.pt`, `checkpoint_epoch_375.pt`, `checkpoint_epoch_400.pt`, `checkpoint_epoch_425.pt`, `checkpoint_epoch_450.pt`, `checkpoint_epoch_475.pt`, `checkpoint_epoch_50.pt`, `checkpoint_epoch_500.pt`, `checkpoint_epoch_525.pt`, `checkpoint_epoch_550.pt`, `checkpoint_epoch_575.pt`, `checkpoint_epoch_600.pt`, `checkpoint_epoch_75.pt`, `last_model.pt`
