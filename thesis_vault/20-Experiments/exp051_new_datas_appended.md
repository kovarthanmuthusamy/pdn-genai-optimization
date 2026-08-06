---
title: exp051_new_datas_appended
type: experiment
status: historical
era: vae
tags: [experiment, exp051_new_datas_appended, historical, era-vae]
---

# exp051_new_datas_appended

**Lineage:** [[exp050]] → **exp051_new_datas_appended** → [[exp052_unbounded_pearson]]
**Status:** historical · **Era:** VAE era (17 Python files)

## Key hyperparameters

| Key | Value |
|-----|-------|
| `latent_dim` | `65` |
| `heatmap_private_dim` | `15` |
| `cond_dim` | `8` |
| `freq_fourier_features` | `8` |
| `num_epochs` | `950` |
| `batch_size` | `160` |
| `learning_rate` | `2e-05` |
| `train_split` | `0.9` |
| `split_by_design` | `True` |
| `layout_train_prob` | `0.55` |
| `heatmap_weight` | `5.5` |
| `impedance_weight` | `1.25` |
| `occupancy_weight` | `7.0` |
| `cross_freq_weight` | `1.0` |
| `data_dir` | `datasets/data_multifreq_train_norm_robust` |
| `resume_checkpoint` | `experiments/exp051_new_datas_appended/checkpoints/checkpoint_epoch_850.pt` |

*Full config: `experiments/exp051_new_datas_appended/config.yaml` (186 keys)*

## Notes (from `experiments/exp051_new_datas_appended/notes.md`)

# Experiment: exp051_new_datas_appended

## Goal

Fine-tune the exp050 Tier-A VAE on the expanded multifreq training set
(`datasets/data_multifreq_train_norm_robust`, ~472k manifest rows including
combinations-appended layouts).

## Changes vs exp050

- `balance_k=false`, `balance_freq=false` — dataset already inverse-K sampled and 16-anchor uniform; avoid double reweighting

- `data_dir` → `datasets/data_multifreq_train_norm_robust` (robust per-MHz log norm)
- Resume from `experiments/exp050/checkpoints/checkpoint_epoch_700.pt`
- Train epochs 701–1000 (`recalculate_curriculum_on_resume=true`)
- All training/eval imports point at `experiments.exp051_new_datas_appended.codes`

## Run

```bash
cd /home/ubuntu/genai_pdn
.venv/bin/python -m experiments.exp051_new_datas_appended.codes.train_vae_simple
# or GPU 1:
./experiments/exp051_new_datas_appended/run_train_gpu1.sh
```

## Results

(TBD)

## Decision

(TBD)

## Code modules

- [[experiments.exp051_new_datas_appended.__init__]]
- [[experiments.exp051_new_datas_appended.codes.__init__]]
- [[experiments.exp051_new_datas_appended.codes.dataloader_multifreq]] — exp051 multifreq dataloader — high-MHz cross-freq bias + append-tag curriculum sampling.
- [[experiments.exp051_new_datas_appended.codes.eval_real_data_sweep]] — Evaluate exp046 heatmap quality using REAL dataset layouts.
- [[experiments.exp051_new_datas_appended.codes.eval_spatial_metrics]] — Off-anchor eval with spatial metrics (Pearson r, peak location) + early-stop hook.
- [[experiments.exp051_new_datas_appended.codes.eval_train_vs_val]] — Diagnostic: compare layout_cross vs encode_cross on TRAIN vs VAL designs.
- [[exp051_eval_common]] — Shared paths, model loading, and PI_freq helpers for exp051 evaluation scripts.
- [[experiments.exp051_new_datas_appended.codes.heatmap_peak_losses]] — Tier A heatmap losses for exp051.
- [[experiments.exp051_new_datas_appended.codes.inference_vae]] — Inference script for Multi-Input VAE — exp051.
- [[experiments.exp051_new_datas_appended.codes.run_epoch_encode]] — Training epoch loop for exp051 — pixel peak blob losses + layout path.
- [[experiments.exp051_new_datas_appended.codes.sampler_curriculum]] — Per-epoch train sampling and layout-path curriculum for exp051.
- [[experiments.exp051_new_datas_appended.codes.spatial_metrics]] — Foreground spatial metrics — Pearson correlation and soft peak location.
- [[experiments.exp051_new_datas_appended.codes.synthetic_freq_blend]] — Synthetic between-anchor heatmap targets for multifreq training (exp045).
- [[experiments.exp051_new_datas_appended.codes.train_vae_simple]] — Train VAE for exp051 — pixel peak blob losses, no percentile training terms.
- [[experiments.exp051_new_datas_appended.codes.vae_multi_input_simple]]
- [[experiments.exp051_new_datas_appended.codes.vae_poe_freq]] — Multi-input VAE — exp045: U-Net skips + freq PoE private dims + encode-first.
- [[experiments.exp051_new_datas_appended.codes.visualize_latent]] — visualize_latent.py — Latent space visualizations for exp051_new_datas_appended (PI_freq-conditioned VAE).

## Metrics artifacts

`experiments/exp051_new_datas_appended/metrics/`

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
- `off_anchor_eval_epoch_50.csv`
- `off_anchor_eval_epoch_75.csv`
- `timing.json`

## Checkpoints

`checkpoint_epoch_1.pt`, `checkpoint_epoch_100.pt`, `checkpoint_epoch_125.pt`, `checkpoint_epoch_150.pt`, `checkpoint_epoch_175.pt`, `checkpoint_epoch_200.pt`, `checkpoint_epoch_225.pt`, `checkpoint_epoch_25.pt`, `checkpoint_epoch_50.pt`, `checkpoint_epoch_75.pt`
