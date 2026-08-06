---
title: exp055_hard_occ
type: experiment
status: historical
era: vae
tags: [experiment, exp055_hard_occ, historical, era-vae]
---

# exp055_hard_occ

**Lineage:** [[exp054_K_30]] → **exp055_hard_occ** → [[exp056_graph_vae]]
**Status:** historical · **Era:** VAE era (20 Python files)

## Key hyperparameters

| Key | Value |
|-----|-------|
| `latent_dim` | `65` |
| `heatmap_private_dim` | `15` |
| `cond_dim` | `8` |
| `freq_fourier_features` | `8` |
| `num_epochs` | `400` |
| `batch_size` | `224` |
| `learning_rate` | `3e-05` |
| `train_split` | `0.9` |
| `split_by_design` | `True` |
| `layout_train_prob` | `0.55` |
| `occupancy_binary_decode` | `True` |
| `heatmap_weight` | `4.0` |
| `impedance_weight` | `4.0` |
| `occupancy_weight` | `7.0` |
| `cross_freq_weight` | `1.0` |
| `data_dir` | `datasets/data_multifreq_train_norm_unbounded` |
| `resume_checkpoint` | `experiments/exp055_hard_occ/checkpoints/checkpoint_epoch_50.pt` |

*Full config: `experiments/exp055_hard_occ/config.yaml` (172 keys)*

## Notes (from `experiments/exp055_hard_occ/notes.md`)

# Experiment: exp055_hard_occ

## Goal
exp054 successor: **binary occupancy before heatmap decode** so inference/QC matches CAD discrete layouts.
Fresh 400-epoch training on unified multifreq dataset (K≈30 focus).

## vs exp054
| Change | exp054 | exp055 |
|--------|--------|--------|
| Decode occupancy | soft probs OK in decoder cond | **top-K binary** before decode (`occupancy_binary_decode`) |
| Cross-freq encode skips | off | off (same) |
| Training | 800ep resume path | **fresh 400ep** |

## Binary occupancy
- `occupancy_binary.py`: `topk_occ_binary`, `occupancy_for_heatmap_decode`
- Model `decode()` binarizes when `binary_occupancy_decode=True` (default)
- Off-anchor eval uses same binary occ for encode_cross + layout_cross
- PEB/CAD sim should use the **same** binary vector as decode conditioning

## Schedule (400 epochs)
| Epoch | Event |
|------:|-------|
| 20 | Cross-freq loss on |
| 80 | Impedance peak loss ramp |
| 260 | Heatmap focus phase |

## Run
```bash
./experiments/exp055_hard_occ/run_train_gpu1.sh
# or DDP:
./experiments/exp055_hard_occ/run_train_ddp.sh
```

## Results
(TBD)

## Code modules

- [[experiments.exp055_hard_occ.codes.__init__]]
- [[experiments.exp055_hard_occ.codes.dataloader_base]] — Multi-frequency PI heatmap dataloader (vendored for exp055).
- [[experiments.exp055_hard_occ.codes.dataloader_multifreq]] — exp055 multifreq dataloader — cross-freq pairs + optional K/freq balance.
- [[experiments.exp055_hard_occ.codes.diagnose_nan_grad]] — Isolate which loss term / layer produces non-finite gradients (exp052).
- [[experiments.exp055_hard_occ.codes.distributed_train]] — DDP helpers for exp055 multi-GPU training.
- [[experiments.exp055_hard_occ.codes.eval_off_anchor]] — Off-anchor eval hook for exp055 training checkpoints.
- [[experiments.exp055_hard_occ.codes.eval_spatial_metrics]] — Off-anchor eval — spatial metrics, append rows to one CSV (epoch column).
- [[exp055_common]] — exp055 shared paths, yaml config, and VAE constructor kwargs.
- [[experiments.exp055_hard_occ.codes.heatmap_peak_losses]] — Heatmap losses: Pearson+grad tier_a + log1p peak/valley extrema.
- [[experiments.exp055_hard_occ.codes.impedance_spectrum_loss]] — Impedance spectrum loss for exp055 — lean stack, no redundant terms.
- [[experiments.exp055_hard_occ.codes.inference_vae]] — Inference script for Multi-Input VAE — exp052.
- [[experiments.exp055_hard_occ.codes.occupancy_binary]] — Binary occupancy for heatmap decode — matches CAD / ECADStar discrete layouts.
- [[experiments.exp055_hard_occ.codes.physics_loss]] — Physics-informed loss modules for exp031 Multi-Input VAE.
- [[experiments.exp055_hard_occ.codes.run_epoch_encode]] — exp055 training epoch — layout path + log1p peak/valley.
- [[experiments.exp055_hard_occ.codes.spatial_metrics]] — FG Pearson + soft extrema location metrics.
- [[experiments.exp055_hard_occ.codes.train_core]] — exp055 training core loop (self-contained).
- [[experiments.exp055_hard_occ.codes.train_vae_simple]] — Train VAE for exp055 — tier_a + peak/valley extrema losses.
- [[experiments.exp055_hard_occ.codes.training_guard]] — Finite-loss / NaN guards for exp055 training.
- [[experiments.exp055_hard_occ.codes.vae_multi_input_simple]]
- [[experiments.exp055_hard_occ.codes.vae_poe_freq]] — Multi-input VAE — exp045: U-Net skips + freq PoE private dims + encode-first.

## Metrics artifacts

`experiments/exp055_hard_occ/metrics/`

- `epoch_timing.csv`
- `heatmap_peak_split.csv`
- `impedance_split.csv`
- `loss.csv`
- `off_anchor_eval.csv`
- `timing.json`

## Checkpoints

`checkpoint_epoch_1.pt`, `checkpoint_epoch_100.pt`, `checkpoint_epoch_125.pt`, `checkpoint_epoch_150.pt`, `checkpoint_epoch_175.pt`, `checkpoint_epoch_200.pt`, `checkpoint_epoch_225.pt`, `checkpoint_epoch_25.pt`, `checkpoint_epoch_250.pt`, `checkpoint_epoch_275.pt`, `checkpoint_epoch_300.pt`, `checkpoint_epoch_325.pt`, `checkpoint_epoch_350.pt`, `checkpoint_epoch_375.pt`, `checkpoint_epoch_400.pt`, `checkpoint_epoch_50.pt`, `checkpoint_epoch_75.pt`, `last_model.pt`
