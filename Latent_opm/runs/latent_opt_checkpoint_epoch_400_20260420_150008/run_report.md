# Latent Optimisation Run Report

**Run:** `latent_opt_checkpoint_epoch_400_20260420_150008`  
**Started:** 20260420_150008  
**Device:** cuda  
**Total runtime:** 14m 16s  
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
| 1 | batch | 21.0 s |
| 2 | batch | 18.5 s |
| 3 | batch | 17.9 s |
| 4 | batch | 16.8 s |
| 5 | batch | 16.4 s |
| 6 | batch | 16.3 s |
| 7 | batch | 16.4 s |
| 8 | batch | 17.0 s |
| 9 | batch | 16.3 s |
| 10 | batch | 16.0 s |
| 11 | batch | 16.4 s |
| 12 | batch | 16.4 s |
| 13 | batch | 16.5 s |
| 14 | batch | 16.5 s |
| 15 | batch | 16.6 s |
| 16 | batch | 17.3 s |
| 17 | batch | 16.7 s |
| 18 | batch | 16.4 s |
| 19 | batch | 16.5 s |
| 20 | batch | 16.3 s |
| 21 | batch | 16.3 s |
| 22 | batch | 16.1 s |
| 23 | batch | 16.3 s |
| 24 | batch | 16.6 s |
| 25 | batch | 16.5 s |
| 26 | batch | 16.4 s |
| 27 | batch | 15.9 s |
| 28 | batch | 16.6 s |
| 29 | batch | 16.6 s |
| 30 | batch | 16.1 s |
| 31 | batch | 16.3 s |
| 32 | batch | 16.5 s |
| 33 | batch | 16.6 s |
| 34 | batch | 16.1 s |
| 35 | batch | 16.7 s |
| 36 | batch | 16.3 s |
| 37 | batch | 16.2 s |
| 38 | batch | 16.2 s |
| 39 | batch | 16.4 s |
| 40 | batch | 16.6 s |
| 41 | batch | 15.9 s |
| 42 | batch | 15.9 s |
| 43 | batch | 15.8 s |
| 44 | batch | 16.1 s |
| 45 | batch | 15.6 s |
| 46 | batch | 16.0 s |
| 47 | batch | 15.8 s |
| 48 | batch | 15.5 s |
| 49 | batch | 15.7 s |
| 50 | batch | 15.6 s |
| 51 | batch | 15.7 s |
| 52 | batch | 15.6 s |
