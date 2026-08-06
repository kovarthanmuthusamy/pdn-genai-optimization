---
title: exp057_structured_graph
type: experiment
status: legacy
era: vae
tags: [experiment, exp057_structured_graph, legacy, era-vae]
---

# exp057_structured_graph

**Lineage:** [[exp056_graph_vae]] → **exp057_structured_graph** → [[exp058_asymmetric_kl]]
**Status:** legacy · **Era:** VAE era (22 Python files)

## Key hyperparameters

| Key | Value |
|-----|-------|
| `latent_dim` | `65` |
| `heatmap_private_dim` | `15` |
| `cond_dim` | `8` |
| `freq_fourier_features` | `8` |
| `num_epochs` | `1450` |
| `batch_size` | `224` |
| `learning_rate` | `1.5e-05` |
| `train_split` | `0.9` |
| `split_by_design` | `True` |
| `layout_train_prob` | `0.65` |
| `occ_only_encode_prob` | `0.4` |
| `occupancy_binary_decode` | `True` |
| `heatmap_weight` | `5.0` |
| `impedance_weight` | `4.0` |
| `occupancy_weight` | `7.0` |
| `cross_freq_weight` | `1.2` |
| `data_dir` | `datasets/data_multifreq_train_norm_unbounded` |
| `al_overlay_data_dir` | `datasets/data_multifreq_al_overlay_exp057` |
| `resume_checkpoint` | `experiments/exp057_structured_graph/checkpoints/last_model.pt` |

*Full config: `experiments/exp057_structured_graph/config.yaml` (181 keys)*

## Config variants

- `config_al_finetune.runtime.yaml`
- `config_al_finetune.yaml`
- `config_occ_only_finetune.runtime.yaml`
- `config_occ_only_finetune.yaml`

## Notes (from `experiments/exp057_structured_graph/notes.md`)

# Experiment: exp056_hard_occ

## Goal
exp054 successor: **binary occupancy before heatmap decode** so inference/QC matches CAD discrete layouts.
Fresh 400-epoch training on unified multifreq dataset (K≈30 focus).

## vs exp054
| Change | exp054 | exp056 |
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
./experiments/exp056_hard_occ/run_train_gpu1.sh
# or DDP:
./experiments/exp056_hard_occ/run_train_ddp.sh
```

## Results
(TBD)

## Code modules

- [[experiments.exp057_structured_graph.codes.__init__]]
- [[experiments.exp057_structured_graph.codes.dataloader_base]] — Multi-frequency PI heatmap dataloader (vendored for exp055).
- [[experiments.exp057_structured_graph.codes.dataloader_multifreq]] — exp055 multifreq dataloader — cross-freq pairs + optional K/freq balance.
- [[experiments.exp057_structured_graph.codes.diagnose_nan_grad]] — Isolate which loss term / layer produces non-finite gradients (exp052).
- [[experiments.exp057_structured_graph.codes.distributed_train]] — DDP helpers for exp055 multi-GPU training.
- [[experiments.exp057_structured_graph.codes.eval_off_anchor]] — Off-anchor eval hook for exp055 training checkpoints.
- [[experiments.exp057_structured_graph.codes.eval_spatial_metrics]] — Off-anchor eval — spatial metrics, append rows to one CSV (epoch column).
- [[exp057_common]] — exp057 shared paths, yaml config, and VAE constructor kwargs.
- [[experiments.exp057_structured_graph.codes.graph_imp]] — 1D spectrum GNN for PI impedance (231 bins) — exp057.
- [[experiments.exp057_structured_graph.codes.graph_occ]] — Graph message-passing occupancy encoder/decoder for exp056 Graph VAE.
- [[experiments.exp057_structured_graph.codes.heatmap_peak_losses]] — Heatmap losses: Pearson+grad tier_a + log1p peak/valley extrema.
- [[experiments.exp057_structured_graph.codes.impedance_spectrum_loss]] — Impedance spectrum loss for exp055 — lean stack, no redundant terms.
- [[experiments.exp057_structured_graph.codes.inference_vae]] — Inference script for Multi-Input VAE — exp052.
- [[experiments.exp057_structured_graph.codes.occupancy_binary]] — Binary occupancy for heatmap decode — matches CAD / ECADStar discrete layouts.
- [[experiments.exp057_structured_graph.codes.physics_loss]] — Physics-informed loss modules for exp031 Multi-Input VAE.
- [[experiments.exp057_structured_graph.codes.run_epoch_encode]] — exp055 training epoch — layout path + log1p peak/valley.
- [[experiments.exp057_structured_graph.codes.spatial_metrics]] — FG Pearson + soft extrema location metrics.
- [[experiments.exp057_structured_graph.codes.train_core]] — exp055 training core loop (self-contained).
- [[experiments.exp057_structured_graph.codes.train_vae_simple]] — Train VAE for exp057 Graph VAE — tier_a + peak/valley extrema losses.
- [[experiments.exp057_structured_graph.codes.training_guard]] — Finite-loss / NaN guards for exp055 training.
- [[experiments.exp057_structured_graph.codes.vae_multi_input_simple]]
- [[experiments.exp057_structured_graph.codes.vae_poe_freq]] — Multi-input VAE — exp045: U-Net skips + freq PoE private dims + encode-first.

## Metrics artifacts

`experiments/exp057_structured_graph/metrics/`

- `epoch_timing.csv`
- `heatmap_peak_split.csv`
- `impedance_split.csv`
- `loss.csv`
- `off_anchor_eval.csv`
- `timing.json`

## Checkpoints

`best_off_anchor_model.pt`, `checkpoint_epoch_1.pt`, `checkpoint_epoch_100.pt`, `checkpoint_epoch_1000.pt`, `checkpoint_epoch_1025.pt`, `checkpoint_epoch_1050.pt`, `checkpoint_epoch_1075.pt`, `checkpoint_epoch_1100.pt`, `checkpoint_epoch_1125.pt`, `checkpoint_epoch_1150.pt`, `checkpoint_epoch_1175.pt`, `checkpoint_epoch_1200.pt`, `checkpoint_epoch_1225.pt`, `checkpoint_epoch_125.pt`, `checkpoint_epoch_1250.pt`, `checkpoint_epoch_1255.pt`, `checkpoint_epoch_1260.pt`, `checkpoint_epoch_1265.pt`, `checkpoint_epoch_1270.pt`, `checkpoint_epoch_1275.pt`, `checkpoint_epoch_1280.pt`, `checkpoint_epoch_1285.pt`, `checkpoint_epoch_1290.pt`, `checkpoint_epoch_1295.pt`, `checkpoint_epoch_1300.pt`, `checkpoint_epoch_1305.pt`, `checkpoint_epoch_1310.pt`, `checkpoint_epoch_1315.pt`, `checkpoint_epoch_1320.pt`, `checkpoint_epoch_1325.pt`, `checkpoint_epoch_1330.pt`, `checkpoint_epoch_1335.pt`, `checkpoint_epoch_1340.pt`, `checkpoint_epoch_1345.pt`, `checkpoint_epoch_1350.pt`, `checkpoint_epoch_1355.pt`, `checkpoint_epoch_1360.pt`, `checkpoint_epoch_1365.pt`, `checkpoint_epoch_1370.pt`, `checkpoint_epoch_1375.pt`, `checkpoint_epoch_1380.pt`, `checkpoint_epoch_1385.pt`, `checkpoint_epoch_1390.pt`, `checkpoint_epoch_1395.pt`, `checkpoint_epoch_1400.pt`, `checkpoint_epoch_1405.pt`, `checkpoint_epoch_1410.pt`, `checkpoint_epoch_1415.pt`, `checkpoint_epoch_1420.pt`, `checkpoint_epoch_1425.pt`, `checkpoint_epoch_1430.pt`, `checkpoint_epoch_1435.pt`, `checkpoint_epoch_150.pt`, `checkpoint_epoch_175.pt`, `checkpoint_epoch_200.pt`, `checkpoint_epoch_225.pt`, `checkpoint_epoch_25.pt`, `checkpoint_epoch_250.pt`, `checkpoint_epoch_275.pt`, `checkpoint_epoch_300.pt`, `checkpoint_epoch_325.pt`, `checkpoint_epoch_350.pt`, `checkpoint_epoch_375.pt`, `checkpoint_epoch_400.pt`, `checkpoint_epoch_425.pt`, `checkpoint_epoch_450.pt`, `checkpoint_epoch_475.pt`, `checkpoint_epoch_50.pt`, `checkpoint_epoch_500.pt`, `checkpoint_epoch_525.pt`, `checkpoint_epoch_550.pt`, `checkpoint_epoch_575.pt`, `checkpoint_epoch_600.pt`, `checkpoint_epoch_625.pt`, `checkpoint_epoch_650.pt`, `checkpoint_epoch_675.pt`, `checkpoint_epoch_700.pt`, `checkpoint_epoch_725.pt`, `checkpoint_epoch_75.pt`, `checkpoint_epoch_750.pt`, `checkpoint_epoch_775.pt`, `checkpoint_epoch_800.pt`, `checkpoint_epoch_825.pt`, `checkpoint_epoch_850.pt`, `checkpoint_epoch_875.pt`, `checkpoint_epoch_900.pt`, `checkpoint_epoch_925.pt`, `checkpoint_epoch_950.pt`, `checkpoint_epoch_975.pt`, `last_model.pt`
