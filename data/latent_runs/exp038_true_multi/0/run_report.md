# Latent Optimisation Run Report

**Experiment:** `exp038_true_multi`  
**Run:** `0` (index 0)  
**Started:** 2026-05-25T11:41:25  
**Device:** cuda  
**Total runtime:** 2m 45s  
**Checkpoint:** `/home/ubuntu/gan/experiments/exp038_true_multi/checkpoints/last_model.pt`  
**Norm stats:** `/home/ubuntu/gan/datasets/data_multifreq_norm/normalization_stats.json`  

**Seed mode:** `random`  
**Master seed:** 1653544593  

## Model

| Parameter | Value |
|-----------|-------|
| `latent_dim` | 42 |
| `cond_dim` | — |
| `heatmap_private_dim` | — |

## Search Space

| Parameter | Value |
|-----------|-------|
| K values | `[1, 2, …, 18, 19] (19 items)` |
| Seeds | `[1187121062, 603921613]` |
| Mode | batch (joint + diversity penalty) |

## Optimisation Hyperparameters

| Parameter | Value |
|-----------|-------|
| `num_steps` | 1200 |
| `lr` | 0.05 |
| `grad_clip` | 5.0 |
| `init_shared_temp` | — |
| `loss_space` | `log` |
| `objective_mode` | `gap_max` |

## Loss & Regularisation Weights

| Weight | Value |
|--------|-------|
| `z_l2_weight` | 0.05 |
| `z_prior_mode` | `—` |
| `boundary_margin` | — |
| `gap_reward_weight` | — |
| `exceed_weight` | — |
| `exceed_power` | — |
| `shape_reg_weight` | — |
| `shape_min_std_dlog` | — |
| `chan_rough_weight` | — |
| `posterior_boundary_weight` | — |
| `posterior_sigma_limit` | — |
| `diversity_weight` | — |
| `occ_confidence_weight` | — |

## Normalisation Statistics (impedance log)

| Stat | Value |
|------|-------|
| `imp_log_mean` | -1.1636513471603394 |
| `imp_log_std` | 1.8059486150741577 |

## Per-K Timing

| K | Seconds | Solution | max_ohm | winning_seed |
|---|---------|----------|---------|--------------|
| 1 | 9.3 s | yes | 9.432 | 603921613 |
| 2 | 8.1 s | yes | 7.649 | 1187121062 |
| 3 | 7.9 s | yes | 8.748 | 1187121062 |
| 4 | 7.6 s | yes | 8.707 | 603921613 |
| 5 | 7.8 s | yes | 8.816 | 1187121062 |
| 6 | 7.8 s | yes | 8.452 | 603921613 |
| 7 | 8.4 s | yes | 8.396 | 603921613 |
| 8 | 10.5 s | yes | 8.135 | 1187121062 |
| 9 | 10.0 s | yes | 7.403 | 1187121062 |
| 10 | 8.7 s | yes | 7.119 | 1187121062 |
| 11 | 10.0 s | yes | 7.083 | 1187121062 |
| 12 | 9.6 s | yes | 7.296 | 1187121062 |
| 13 | 9.2 s | yes | 7.337 | 1187121062 |
| 14 | 9.0 s | yes | 7.526 | 1187121062 |
| 15 | 9.7 s | yes | 8.526 | 1187121062 |
| 16 | 8.8 s | yes | 8.918 | 1187121062 |
| 17 | 7.2 s | yes | 8.795 | 1187121062 |
| 18 | 7.1 s | yes | 8.52 | 1187121062 |
| 19 | 7.2 s | yes | 9.057 | 1187121062 |

## No Solution Found

Every K in the run produced a saved `best_*` solution (or the run is incomplete).

## Results Summary

| K | Solution | max_ohm | best_score | Step | Seed |
|---|----------|---------|------------|------|------|
| 1 | yes | 9.432 | 9.432 | 116 | 603921613 |
| 10 | yes | 7.119 | 7.119 | 19 | 1187121062 |
| 11 | yes | 7.083 | 7.083 | 30 | 1187121062 |
| 12 | yes | 7.296 | 7.296 | 6 | 1187121062 |
| 13 | yes | 7.337 | 7.337 | 5 | 1187121062 |
| 14 | yes | 7.526 | 7.526 | 5 | 1187121062 |
| 15 | yes | 8.526 | 8.526 | 5 | 1187121062 |
| 16 | yes | 8.918 | 8.918 | 4 | 1187121062 |
| 17 | yes | 8.795 | 8.795 | 18 | 1187121062 |
| 18 | yes | 8.52 | 8.52 | 79 | 1187121062 |
| 19 | yes | 9.057 | 9.057 | 47 | 1187121062 |
| 2 | yes | 7.649 | 7.649 | 242 | 1187121062 |
| 3 | yes | 8.748 | 8.748 | 222 | 1187121062 |
| 4 | yes | 8.707 | 8.707 | 16 | 603921613 |
| 5 | yes | 8.816 | 8.816 | 717 | 1187121062 |
| 6 | yes | 8.452 | 8.452 | 216 | 603921613 |
| 7 | yes | 8.396 | 8.396 | 239 | 603921613 |
| 8 | yes | 8.135 | 8.135 | 7 | 1187121062 |
| 9 | yes | 7.403 | 7.403 | 14 | 1187121062 |

**K with saved solution:** **19/19**  

## Optimization Plots

### K = 01

#### Impedance

![K01 Impedance](K01/latent_opt_impedance.png)

#### Heatmaps

![K01 Heatmaps](K01/latent_opt_heatmaps.png)

#### Occupancy

![K01 Occupancy](K01/latent_opt_occupancy.png)

### K = 02

#### Impedance

![K02 Impedance](K02/latent_opt_impedance.png)

#### Heatmaps

![K02 Heatmaps](K02/latent_opt_heatmaps.png)

#### Occupancy

![K02 Occupancy](K02/latent_opt_occupancy.png)

### K = 03

#### Impedance

![K03 Impedance](K03/latent_opt_impedance.png)

#### Heatmaps

![K03 Heatmaps](K03/latent_opt_heatmaps.png)

#### Occupancy

![K03 Occupancy](K03/latent_opt_occupancy.png)

### K = 04

#### Impedance

![K04 Impedance](K04/latent_opt_impedance.png)

#### Heatmaps

![K04 Heatmaps](K04/latent_opt_heatmaps.png)

#### Occupancy

![K04 Occupancy](K04/latent_opt_occupancy.png)

### K = 05

#### Impedance

![K05 Impedance](K05/latent_opt_impedance.png)

#### Heatmaps

![K05 Heatmaps](K05/latent_opt_heatmaps.png)

#### Occupancy

![K05 Occupancy](K05/latent_opt_occupancy.png)

### K = 06

#### Impedance

![K06 Impedance](K06/latent_opt_impedance.png)

#### Heatmaps

![K06 Heatmaps](K06/latent_opt_heatmaps.png)

#### Occupancy

![K06 Occupancy](K06/latent_opt_occupancy.png)

### K = 07

#### Impedance

![K07 Impedance](K07/latent_opt_impedance.png)

#### Heatmaps

![K07 Heatmaps](K07/latent_opt_heatmaps.png)

#### Occupancy

![K07 Occupancy](K07/latent_opt_occupancy.png)

### K = 08

#### Impedance

![K08 Impedance](K08/latent_opt_impedance.png)

#### Heatmaps

![K08 Heatmaps](K08/latent_opt_heatmaps.png)

#### Occupancy

![K08 Occupancy](K08/latent_opt_occupancy.png)

### K = 09

#### Impedance

![K09 Impedance](K09/latent_opt_impedance.png)

#### Heatmaps

![K09 Heatmaps](K09/latent_opt_heatmaps.png)

#### Occupancy

![K09 Occupancy](K09/latent_opt_occupancy.png)

### K = 10

#### Impedance

![K10 Impedance](K10/latent_opt_impedance.png)

#### Heatmaps

![K10 Heatmaps](K10/latent_opt_heatmaps.png)

#### Occupancy

![K10 Occupancy](K10/latent_opt_occupancy.png)

### K = 11

#### Impedance

![K11 Impedance](K11/latent_opt_impedance.png)

#### Heatmaps

![K11 Heatmaps](K11/latent_opt_heatmaps.png)

#### Occupancy

![K11 Occupancy](K11/latent_opt_occupancy.png)

### K = 12

#### Impedance

![K12 Impedance](K12/latent_opt_impedance.png)

#### Heatmaps

![K12 Heatmaps](K12/latent_opt_heatmaps.png)

#### Occupancy

![K12 Occupancy](K12/latent_opt_occupancy.png)

### K = 13

#### Impedance

![K13 Impedance](K13/latent_opt_impedance.png)

#### Heatmaps

![K13 Heatmaps](K13/latent_opt_heatmaps.png)

#### Occupancy

![K13 Occupancy](K13/latent_opt_occupancy.png)

### K = 14

#### Impedance

![K14 Impedance](K14/latent_opt_impedance.png)

#### Heatmaps

![K14 Heatmaps](K14/latent_opt_heatmaps.png)

#### Occupancy

![K14 Occupancy](K14/latent_opt_occupancy.png)

### K = 15

#### Impedance

![K15 Impedance](K15/latent_opt_impedance.png)

#### Heatmaps

![K15 Heatmaps](K15/latent_opt_heatmaps.png)

#### Occupancy

![K15 Occupancy](K15/latent_opt_occupancy.png)

### K = 16

#### Impedance

![K16 Impedance](K16/latent_opt_impedance.png)

#### Heatmaps

![K16 Heatmaps](K16/latent_opt_heatmaps.png)

#### Occupancy

![K16 Occupancy](K16/latent_opt_occupancy.png)

### K = 17

#### Impedance

![K17 Impedance](K17/latent_opt_impedance.png)

#### Heatmaps

![K17 Heatmaps](K17/latent_opt_heatmaps.png)

#### Occupancy

![K17 Occupancy](K17/latent_opt_occupancy.png)

### K = 18

#### Impedance

![K18 Impedance](K18/latent_opt_impedance.png)

#### Heatmaps

![K18 Heatmaps](K18/latent_opt_heatmaps.png)

#### Occupancy

![K18 Occupancy](K18/latent_opt_occupancy.png)

### K = 19

#### Impedance

![K19 Impedance](K19/latent_opt_impedance.png)

#### Heatmaps

![K19 Heatmaps](K19/latent_opt_heatmaps.png)

#### Occupancy

![K19 Occupancy](K19/latent_opt_occupancy.png)
