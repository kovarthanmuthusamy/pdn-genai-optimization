---
title: exp058_asymmetric_kl
type: experiment
status: legacy
era: vae
tags: [experiment, exp058_asymmetric_kl, legacy, era-vae]
---

# exp058_asymmetric_kl

**Lineage:** [[exp057_structured_graph]] → **exp058_asymmetric_kl** → [[exp059_capacity_freq]]
**Status:** legacy · **Era:** VAE era (22 Python files)

## Key hyperparameters

| Key | Value |
|-----|-------|
| `latent_dim` | `65` |
| `heatmap_private_dim` | `15` |
| `cond_dim` | `8` |
| `freq_fourier_features` | `8` |
| `num_epochs` | `445` |
| `batch_size` | `224` |
| `learning_rate` | `1.5e-05` |
| `train_split` | `0.9` |
| `split_by_design` | `True` |
| `layout_train_prob` | `0.9` |
| `occ_only_encode_prob` | `0.7` |
| `occupancy_binary_decode` | `True` |
| `heatmap_weight` | `5.0` |
| `impedance_weight` | `4.0` |
| `occupancy_weight` | `7.0` |
| `cross_freq_weight` | `1.2` |
| `data_dir` | `datasets/data_multifreq_train_norm_unbounded` |
| `al_overlay_data_dir` | `datasets/data_multifreq_al_overlay_exp058` |
| `resume_checkpoint` | `experiments/exp058_asymmetric_kl/checkpoints/last_model.pt` |

*Full config: `experiments/exp058_asymmetric_kl/config.yaml` (181 keys)*

## Config variants

- `config_al_finetune.runtime.yaml`
- `config_al_finetune.yaml`
- `config_occ_only_finetune.yaml`

## Notes (from `experiments/exp058_asymmetric_kl/notes.md`)

# Experiment: exp058_asymmetric_kl

## Goal
**Fresh train from scratch** (does **not** load exp057 weights) with:

1. Lower KL (`beta=0.03`) + higher `free_bits=0.12`
2. Asymmetric train near AL path (`layout_p=0.9`, `occ_only=0.7` + distill)
3. Same arch as exp057 (`latent_dim=65`) for recipe comparison
4. AL scoring always occ-only; promote via `layout_cross`

## vs exp057
| Knob | exp057 | exp058 |
|------|--------|--------|
| Init | continued / AL finetune | **from scratch** |
| `beta_final` | ~0.05 | **0.03** |
| `free_bits` | 0.08 | **0.12** |
| Train mix | varied | **asymmetric 0.9 / 0.7** |
| LR (base) | finetune-scale later | **3e-5**, 400 epochs |

## Run order
```bash
# 1) Fresh base train (required first)
bash experiments/exp058_asymmetric_kl/run_train_gpu1.sh

# 2) Optional short occ-only align (after last_model.pt exists)
COMMAND=occ-warmup python pipelines/active_learning/run.py

# 3) AL cycles
COMMAND=full python pipelines/active_learning/run.py
```

`checkpoints/` starts empty. Do not copy exp057 `last_model.pt` into this run.

## Code modules

- [[experiments.exp058_asymmetric_kl.codes.__init__]]
- [[experiments.exp058_asymmetric_kl.codes.dataloader_base]] — Multi-frequency PI heatmap dataloader (vendored for exp055).
- [[experiments.exp058_asymmetric_kl.codes.dataloader_multifreq]] — exp055 multifreq dataloader — cross-freq pairs + optional K/freq balance.
- [[experiments.exp058_asymmetric_kl.codes.diagnose_nan_grad]] — Isolate which loss term / layer produces non-finite gradients (exp052).
- [[experiments.exp058_asymmetric_kl.codes.distributed_train]] — DDP helpers for exp055 multi-GPU training.
- [[experiments.exp058_asymmetric_kl.codes.eval_off_anchor]] — Off-anchor eval hook for exp055 training checkpoints.
- [[experiments.exp058_asymmetric_kl.codes.eval_spatial_metrics]] — Off-anchor eval — spatial metrics, append rows to one CSV (epoch column).
- [[exp058_common]] — exp058 shared paths, yaml config, and VAE constructor kwargs.
- [[experiments.exp058_asymmetric_kl.codes.graph_imp]] — 1D spectrum GNN for PI impedance (231 bins) — exp058.
- [[experiments.exp058_asymmetric_kl.codes.graph_occ]] — Graph message-passing occupancy encoder/decoder for exp056 Graph VAE.
- [[experiments.exp058_asymmetric_kl.codes.heatmap_peak_losses]] — Heatmap losses: Pearson+grad tier_a + log1p peak/valley extrema.
- [[experiments.exp058_asymmetric_kl.codes.impedance_spectrum_loss]] — Impedance spectrum loss for exp055 — lean stack, no redundant terms.
- [[experiments.exp058_asymmetric_kl.codes.inference_vae]] — Inference script for Multi-Input VAE — exp052.
- [[experiments.exp058_asymmetric_kl.codes.occupancy_binary]] — Binary occupancy for heatmap decode — matches CAD / ECADStar discrete layouts.
- [[experiments.exp058_asymmetric_kl.codes.physics_loss]] — Physics-informed loss modules for exp031 Multi-Input VAE.
- [[experiments.exp058_asymmetric_kl.codes.run_epoch_encode]] — exp055 training epoch — layout path + log1p peak/valley.
- [[experiments.exp058_asymmetric_kl.codes.spatial_metrics]] — FG Pearson + soft extrema location metrics.
- [[experiments.exp058_asymmetric_kl.codes.train_core]] — exp055 training core loop (self-contained).
- [[experiments.exp058_asymmetric_kl.codes.train_vae_simple]] — Train VAE for exp058 Graph VAE — tier_a + peak/valley extrema losses.
- [[experiments.exp058_asymmetric_kl.codes.training_guard]] — Finite-loss / NaN guards for exp055 training.
- [[experiments.exp058_asymmetric_kl.codes.vae_multi_input_simple]]
- [[experiments.exp058_asymmetric_kl.codes.vae_poe_freq]] — Multi-input VAE — exp045: U-Net skips + freq PoE private dims + encode-first.

## Metrics artifacts

`experiments/exp058_asymmetric_kl/metrics/`

- `epoch_timing.csv`
- `heatmap_peak_split.csv`
- `impedance_split.csv`
- `loss.csv`
- `off_anchor_eval.csv`
- `timing.json`

## Checkpoints

`best_off_anchor_model.pt`, `checkpoint_epoch_1.pt`, `checkpoint_epoch_10.pt`, `checkpoint_epoch_100.pt`, `checkpoint_epoch_105.pt`, `checkpoint_epoch_110.pt`, `checkpoint_epoch_115.pt`, `checkpoint_epoch_120.pt`, `checkpoint_epoch_125.pt`, `checkpoint_epoch_130.pt`, `checkpoint_epoch_135.pt`, `checkpoint_epoch_140.pt`, `checkpoint_epoch_145.pt`, `checkpoint_epoch_15.pt`, `checkpoint_epoch_150.pt`, `checkpoint_epoch_155.pt`, `checkpoint_epoch_160.pt`, `checkpoint_epoch_165.pt`, `checkpoint_epoch_170.pt`, `checkpoint_epoch_175.pt`, `checkpoint_epoch_180.pt`, `checkpoint_epoch_185.pt`, `checkpoint_epoch_190.pt`, `checkpoint_epoch_195.pt`, `checkpoint_epoch_20.pt`, `checkpoint_epoch_200.pt`, `checkpoint_epoch_205.pt`, `checkpoint_epoch_210.pt`, `checkpoint_epoch_215.pt`, `checkpoint_epoch_220.pt`, `checkpoint_epoch_225.pt`, `checkpoint_epoch_230.pt`, `checkpoint_epoch_235.pt`, `checkpoint_epoch_240.pt`, `checkpoint_epoch_245.pt`, `checkpoint_epoch_25.pt`, `checkpoint_epoch_250.pt`, `checkpoint_epoch_255.pt`, `checkpoint_epoch_260.pt`, `checkpoint_epoch_265.pt`, `checkpoint_epoch_270.pt`, `checkpoint_epoch_275.pt`, `checkpoint_epoch_280.pt`, `checkpoint_epoch_285.pt`, `checkpoint_epoch_290.pt`, `checkpoint_epoch_295.pt`, `checkpoint_epoch_30.pt`, `checkpoint_epoch_300.pt`, `checkpoint_epoch_305.pt`, `checkpoint_epoch_310.pt`, `checkpoint_epoch_315.pt`, `checkpoint_epoch_320.pt`, `checkpoint_epoch_325.pt`, `checkpoint_epoch_330.pt`, `checkpoint_epoch_335.pt`, `checkpoint_epoch_340.pt`, `checkpoint_epoch_345.pt`, `checkpoint_epoch_35.pt`, `checkpoint_epoch_350.pt`, `checkpoint_epoch_355.pt`, `checkpoint_epoch_360.pt`, `checkpoint_epoch_365.pt`, `checkpoint_epoch_370.pt`, `checkpoint_epoch_375.pt`, `checkpoint_epoch_380.pt`, `checkpoint_epoch_385.pt`, `checkpoint_epoch_390.pt`, `checkpoint_epoch_395.pt`, `checkpoint_epoch_40.pt`, `checkpoint_epoch_400.pt`, `checkpoint_epoch_405.pt`, `checkpoint_epoch_410.pt`, `checkpoint_epoch_415.pt`, `checkpoint_epoch_420.pt`, `checkpoint_epoch_425.pt`, `checkpoint_epoch_45.pt`, `checkpoint_epoch_5.pt`, `checkpoint_epoch_50.pt`, `checkpoint_epoch_55.pt`, `checkpoint_epoch_60.pt`, `checkpoint_epoch_65.pt`, `checkpoint_epoch_70.pt`, `checkpoint_epoch_75.pt`, `checkpoint_epoch_80.pt`, `checkpoint_epoch_85.pt`, `checkpoint_epoch_90.pt`, `checkpoint_epoch_95.pt`, `last_model.pt`
