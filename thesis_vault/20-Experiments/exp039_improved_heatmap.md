---
title: exp039_improved_heatmap
type: experiment
status: historical
era: vae
tags: [experiment, exp039_improved_heatmap, historical, era-vae]
---

# exp039_improved_heatmap

**Lineage:** [[exp038_true_multi]] → **exp039_improved_heatmap** → [[exp040]]
**Status:** historical · **Era:** VAE era (26 Python files)

## Key hyperparameters

| Key | Value |
|-----|-------|
| `latent_dim` | `42` |
| `heatmap_private_dim` | `8` |
| `cond_dim` | `8` |
| `freq_fourier_features` | `8` |
| `num_epochs` | `1050` |
| `batch_size` | `128` |
| `learning_rate` | `4e-05` |
| `train_split` | `0.9` |
| `split_by_design` | `True` |
| `layout_train_prob` | `0.92` |
| `heatmap_weight` | `3.5` |
| `impedance_weight` | `1.5` |
| `occupancy_weight` | `7.0` |
| `cross_freq_weight` | `0.5` |
| `data_dir` | `datasets/data_multifreq_norm` |
| `resume_checkpoint` | `750` |

*Full config: `experiments/exp039_improved_heatmap/config.yaml` (75 keys)*

## Code modules

- [[experiments.exp039_improved_heatmap.codes.__init__]]
- [[experiments.exp039_improved_heatmap.codes.codes.compute_anchor_fg_max]] — Compute per-anchor foreground max (physical Ω) from data_multifreq_norm and write metrics JSON.
- [[experiments.exp039_improved_heatmap.codes.codes.dataloader_multifreq]] — Multi-frequency PI heatmap dataloader for exp038_true_multi.
- [[experiments.exp039_improved_heatmap.codes.codes.eval_cross_freq]] — Evaluate native vs cross-frequency heatmap reconstruction on the val set.
- [[experiments.exp039_improved_heatmap.codes.codes.eval_val_recon]] — Validation reconstruction metrics for exp038 (occ accuracy + impedance peaks).
- [[experiments.exp039_improved_heatmap.codes.codes.evaluate_vae]] — evaluate_vae.py — Post-training evaluation for exp025_latent_size_change
- [[experiments.exp039_improved_heatmap.codes.codes.freq_conditioning]] — PI frequency conditioning: Fourier features + MLP; FiLM for heatmap conv blocks.
- [[experiments.exp039_improved_heatmap.codes.codes.freq_inference_utils]] — PI frequency helpers for inference on unseen / off-anchor MHz (between training anchors).
- [[experiments.exp039_improved_heatmap.codes.codes.impedance_spectrum_loss]] — Peak-aware impedance losses (frequency weighting, dual top-k, peak alignment).
- [[experiments.exp039_improved_heatmap.codes.codes.inference_vae]] — Inference script for Multi-Input VAE — exp038_true_multi.
- [[experiments.exp039_improved_heatmap.codes.codes.metrics_csv_utils]] — Read/write exp038 metrics CSVs (dedupe by epoch, sorted).
- [[experiments.exp039_improved_heatmap.codes.codes.physics_loss]] — Physics-informed loss modules for exp031 Multi-Input VAE.
- [[experiments.exp039_improved_heatmap.codes.codes.refresh_exp038_plots]] — Update metrics/loss.csv (dedupe) and rebuild exp038 plots from VAETrainingLogger.
- [[experiments.exp039_improved_heatmap.codes.codes.save_epoch1_losses]] — Backfill epoch-1 train/val metrics into metrics/loss.csv (no checkpoint write).
- [[experiments.exp039_improved_heatmap.codes.codes.surrogate_impedance]] — Surrogate impedance model and training script — exp038_true_multi.
- [[experiments.exp039_improved_heatmap.codes.codes.train_vae_simple]] — Training — Multi-Input VAE (exp038_true_multi): multifreq PI heatmaps + peak losses.
- [[experiments.exp039_improved_heatmap.codes.codes.vae_multi_input_simple]]
- [[experiments.exp039_improved_heatmap.codes.codes.visualize_latent]] — visualize_latent.py — Latent space visualizations for exp025_latent_size_change
- [[experiments.exp039_improved_heatmap.codes.evaluate_vae]] — evaluate_vae.py — Post-training evaluation for exp039_improved_heatmap (PI_freq-conditioned VAE).
- [[experiments.exp039_improved_heatmap.codes.exp039_eval_common]] — Shared paths, model loading, and PI_freq helpers for exp039 evaluation scripts.
- [[experiments.exp039_improved_heatmap.codes.inference_vae]] — Inference script for Multi-Input VAE — exp039_improved_heatmap.
- [[experiments.exp039_improved_heatmap.codes.metrics_csv_utils]] — Read/write exp039 metrics CSVs (dedupe by epoch, sorted).
- [[experiments.exp039_improved_heatmap.codes.save_epoch1_losses]] — Backfill epoch-1 metrics and rebuild exp039 plots from metrics/loss.csv.
- [[experiments.exp039_improved_heatmap.codes.synthetic_freq_blend]] — Synthetic between-anchor heatmap targets for multifreq training (exp039).
- [[experiments.exp039_improved_heatmap.codes.train_vae_simple]] — Train VAE for exp039_improved_heatmap.
- [[experiments.exp039_improved_heatmap.codes.visualize_latent]] — Latent-space visualizations for exp043 PI_freq-conditioned VAE.

## Metrics artifacts

`experiments/exp039_improved_heatmap/metrics/`

- `epoch_timing.csv`
- `epoch_timing.csv.bak`
- `impedance_split.csv`
- `loss.csv`
- `loss.csv.bak`
- `off_anchor_eval_epoch_100.csv`
- `off_anchor_eval_epoch_1000.csv`
- `off_anchor_eval_epoch_1025.csv`
- `off_anchor_eval_epoch_1050.csv`
- `off_anchor_eval_epoch_125.csv`
- `off_anchor_eval_epoch_150.csv`
- `off_anchor_eval_epoch_175.csv`
- `off_anchor_eval_epoch_200.csv`
- `off_anchor_eval_epoch_225.csv`
- `off_anchor_eval_epoch_25.csv`

## Checkpoints

`checkpoint_epoch_100.pt`, `checkpoint_epoch_1000.pt`, `checkpoint_epoch_1025.pt`, `checkpoint_epoch_1050.pt`, `checkpoint_epoch_125.pt`, `checkpoint_epoch_150.pt`, `checkpoint_epoch_175.pt`, `checkpoint_epoch_200.pt`, `checkpoint_epoch_225.pt`, `checkpoint_epoch_25.pt`, `checkpoint_epoch_250.pt`, `checkpoint_epoch_275.pt`, `checkpoint_epoch_300.pt`, `checkpoint_epoch_325.pt`, `checkpoint_epoch_350.pt`, `checkpoint_epoch_375.pt`, `checkpoint_epoch_400.pt`, `checkpoint_epoch_425.pt`, `checkpoint_epoch_450.pt`, `checkpoint_epoch_475.pt`, `checkpoint_epoch_50.pt`, `checkpoint_epoch_500.pt`, `checkpoint_epoch_525.pt`, `checkpoint_epoch_550.pt`, `checkpoint_epoch_575.pt`, `checkpoint_epoch_600.pt`, `checkpoint_epoch_625.pt`, `checkpoint_epoch_650.pt`, `checkpoint_epoch_675.pt`, `checkpoint_epoch_700.pt`, `checkpoint_epoch_725.pt`, `checkpoint_epoch_75.pt`, `checkpoint_epoch_750.pt`, `checkpoint_epoch_775.pt`, `checkpoint_epoch_800.pt`, `checkpoint_epoch_825.pt`, `checkpoint_epoch_850.pt`, `checkpoint_epoch_875.pt`, `checkpoint_epoch_900.pt`, `checkpoint_epoch_925.pt`, `checkpoint_epoch_950.pt`, `checkpoint_epoch_975.pt`, `last_model.pt`
