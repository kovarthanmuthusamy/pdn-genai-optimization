# Latent Optimisation Run Report

**Run:** `latent_opt_checkpoint_epoch_400_20260420_120119`  
**Started:** 20260420_120119  
**Device:** cuda  
**Total runtime:** 10m 43s  
**Checkpoint:** `experiments/exp030_adding_physic/checkpoints/checkpoint_epoch_400.pt`  

## Model

| Parameter | Value |
|-----------|-------|
| `latent_dim` | 32 |
| `cond_dim` | 8 |
| `heatmap_private_dim` | 8 |

## Search Space

| Parameter | Value |
|-----------|-------|
| K values | `[1, 2, …, 51, 52] (52 items)` |
| Seeds | `[0, 1, …, 30, 31] (32 items)` |
| Mode | batch (joint + diversity penalty) |

## Optimisation Hyperparameters

| Parameter | Value |
|-----------|-------|
| `num_steps` | 1200 |
| `lr` | 0.05 |
| `grad_clip` | 5.0 |
| `init_shared_temp` | 0.8 |
| `loss_space` | `log` |
| `objective_mode` | `gap_max` |

## Loss & Regularisation Weights

| Weight | Value |
|--------|-------|
| `z_l2_weight` | 0.05 |
| `z_prior_mode` | `agg_posterior` |
| `boundary_margin` | 0.0 |
| `gap_reward_weight` | 1.0 |
| `exceed_weight` | 25.0 |
| `exceed_power` | 2.0 |
| `shape_reg_weight` | 2.0 |
| `shape_min_std_dlog` | 0.12 |
| `chan_rough_weight` | 0.3 |
| `posterior_boundary_weight` | 10.0 |
| `posterior_sigma_limit` | 2.5 |
| `diversity_weight` | 0.35 |
| `occ_confidence_weight` | 0.0 |

## Normalisation Statistics

| Stat | Value |
|------|-------|
| `imp_log_mean` | -1.1628280878067017 |
| `imp_log_std` | 1.8058764934539795 |

## Per-K Timing

| K | Mode | Seconds |
|---|------|---------|
| 1 | batch | 15.4 s |
| 2 | batch | 14.4 s |
| 3 | batch | 13.3 s |
| 4 | batch | 12.4 s |
| 5 | batch | 12.6 s |
| 6 | batch | 12.3 s |
| 7 | batch | 12.5 s |
| 8 | batch | 12.5 s |
| 9 | batch | 12.3 s |
| 10 | batch | 11.8 s |
| 11 | batch | 12.6 s |
| 12 | batch | 12.4 s |
| 13 | batch | 12.8 s |
| 14 | batch | 12.3 s |
| 15 | batch | 12.6 s |
| 16 | batch | 12.6 s |
| 17 | batch | 12.4 s |
| 18 | batch | 12.1 s |
| 19 | batch | 12.1 s |
| 20 | batch | 12.2 s |
| 21 | batch | 12.1 s |
| 22 | batch | 12.5 s |
| 23 | batch | 12.2 s |
| 24 | batch | 12.7 s |
| 25 | batch | 12.6 s |
| 26 | batch | 12.6 s |
| 27 | batch | 12.5 s |
| 28 | batch | 12.2 s |
| 29 | batch | 12.3 s |
| 30 | batch | 12.2 s |
| 31 | batch | 12.2 s |
| 32 | batch | 12.5 s |
| 33 | batch | 12.4 s |
| 34 | batch | 11.8 s |
| 35 | batch | 12.2 s |
| 36 | batch | 12.0 s |
| 37 | batch | 12.5 s |
| 38 | batch | 12.5 s |
| 39 | batch | 12.0 s |
| 40 | batch | 12.6 s |
| 41 | batch | 11.8 s |
| 42 | batch | 12.0 s |
| 43 | batch | 12.2 s |
| 44 | batch | 11.6 s |
| 45 | batch | 11.7 s |
| 46 | batch | 12.2 s |
| 47 | batch | 11.8 s |
| 48 | batch | 11.8 s |
| 49 | batch | 11.9 s |
| 50 | batch | 11.9 s |
| 51 | batch | 11.9 s |
| 52 | batch | 11.6 s |
