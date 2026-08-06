---
title: exp052_unbounded_pearson
type: experiment
status: historical
era: vae
tags: [experiment, exp052_unbounded_pearson, historical, era-vae]
---

# exp052_unbounded_pearson

**Lineage:** [[exp051_new_datas_appended]] → **exp052_unbounded_pearson** → [[exp053_peak_log1p_losses]]
**Status:** historical · **Era:** VAE era (19 Python files)

## Key hyperparameters

| Key | Value |
|-----|-------|
| `latent_dim` | `65` |
| `heatmap_private_dim` | `15` |
| `cond_dim` | `8` |
| `freq_fourier_features` | `8` |
| `num_epochs` | `450` |
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
| `resume_checkpoint` | `checkpoints/checkpoint_epoch_175.pt` |

*Full config: `experiments/exp052_unbounded_pearson/config.yaml` (201 keys)*

## Notes (from `experiments/exp052_unbounded_pearson/notes.md`)

# exp052 — unbounded Pearson+grad heatmaps

Fork of exp051 with:

1. **Dataset:** `datasets/data_multifreq_train_norm_unbounded` (`robust_log1p_per_mhz_unbounded`)
2. **Loss:** FG Pearson (global) + Sobel grad (local) + phys top-k blob — no Huber FG, no p99/dynrange
3. **450 epochs** fresh training

Build dataset first:
```bash
python pipelines/normalize/build_train_norm_unbounded.py
```

Train:
```bash
.venv/bin/python -m experiments.exp052_unbounded_pearson.codes.train_vae_simple
```

## Code modules

- [[experiments.exp052_unbounded_pearson.__init__]]
- [[experiments.exp052_unbounded_pearson.codes.__init__]]
- [[experiments.exp052_unbounded_pearson.codes.dataloader_multifreq]] — exp052 multifreq dataloader — high-MHz cross-freq bias + append-tag curriculum sampling.
- [[experiments.exp052_unbounded_pearson.codes.diagnose_nan_grad]] — Isolate which loss term / layer produces non-finite gradients (exp052).
- [[experiments.exp052_unbounded_pearson.codes.eval_real_data_sweep]] — Evaluate exp046 heatmap quality using REAL dataset layouts.
- [[experiments.exp052_unbounded_pearson.codes.eval_spatial_metrics]] — Off-anchor eval with spatial metrics (Pearson r, peak location) + early-stop hook.
- [[experiments.exp052_unbounded_pearson.codes.eval_train_vs_val]] — Diagnostic: compare layout_cross vs encode_cross on TRAIN vs VAL designs.
- [[exp052_eval_common]] — Shared paths, model loading, and PI_freq helpers for exp052 evaluation scripts.
- [[experiments.exp052_unbounded_pearson.codes.heatmap_peak_losses]] — exp052 heatmap losses: FG Pearson (global) + gradient field (local) + log1p peak blob.
- [[experiments.exp052_unbounded_pearson.codes.inference_vae]] — Inference script for Multi-Input VAE — exp052.
- [[experiments.exp052_unbounded_pearson.codes.mhz_loss_weight]] — Per-sample heatmap loss multipliers vs PI frequency (MHz).
- [[experiments.exp052_unbounded_pearson.codes.run_epoch_encode]] — Training epoch loop for exp052 — pixel peak blob losses + layout path.
- [[experiments.exp052_unbounded_pearson.codes.sampler_curriculum]] — Per-epoch train sampling and layout-path curriculum for exp052.
- [[experiments.exp052_unbounded_pearson.codes.spatial_metrics]] — Foreground spatial metrics — Pearson correlation and soft peak location.
- [[experiments.exp052_unbounded_pearson.codes.synthetic_freq_blend]] — Synthetic between-anchor heatmap targets for multifreq training (exp045).
- [[experiments.exp052_unbounded_pearson.codes.train_vae_simple]] — Train VAE for exp052 — Pearson + grad + phys top-k blob (no percentile/dynrange/z-clip).
- [[experiments.exp052_unbounded_pearson.codes.training_guard]] — Finite-loss guards for exp052 training (NaN / Inf prevention).
- [[experiments.exp052_unbounded_pearson.codes.vae_multi_input_simple]]
- [[experiments.exp052_unbounded_pearson.codes.vae_poe_freq]] — Multi-input VAE — exp045: U-Net skips + freq PoE private dims + encode-first.

## Metrics artifacts

`experiments/exp052_unbounded_pearson/metrics/`

- `epoch_timing.csv`
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
- `off_anchor_eval_epoch_325.csv`

## Checkpoints

`checkpoint_epoch_1.pt`, `checkpoint_epoch_100.pt`, `checkpoint_epoch_125.pt`, `checkpoint_epoch_150.pt`, `checkpoint_epoch_175.pt`, `checkpoint_epoch_200.pt`, `checkpoint_epoch_225.pt`, `checkpoint_epoch_25.pt`, `checkpoint_epoch_250.pt`, `checkpoint_epoch_275.pt`, `checkpoint_epoch_300.pt`, `checkpoint_epoch_325.pt`, `checkpoint_epoch_350.pt`, `checkpoint_epoch_375.pt`, `checkpoint_epoch_400.pt`, `checkpoint_epoch_425.pt`, `checkpoint_epoch_450.pt`, `checkpoint_epoch_50.pt`, `checkpoint_epoch_75.pt`, `last_model.pt`
