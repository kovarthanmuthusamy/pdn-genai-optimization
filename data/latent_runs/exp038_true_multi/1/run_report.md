# Latent Optimisation Run Report

**Experiment:** `exp038_true_multi`  
**Run:** `1` (index 1)  
**Started:** 2026-05-25T13:01:25  
**Device:** cuda  
**Total runtime:** 7m 24s  
**Checkpoint:** `/home/ubuntu/gan/experiments/exp038_true_multi/checkpoints/last_model.pt`  
**Norm stats:** `/home/ubuntu/gan/datasets/data_multifreq_norm/normalization_stats.json`  

**Seed mode:** `random`  
**Master seed:** 1399874551  

## Model

| Parameter | Value |
|-----------|-------|
| `latent_dim` | 42 |
| `cond_dim` | — |
| `heatmap_private_dim` | — |

## Search Space

| Parameter | Value |
|-----------|-------|
| K values | `[1, 2, …, 24, 25] (25 items)` |
| Seeds | `[232960483, 512830382, …, 1985357705, 15492134] (32 items)` |
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
| 1 | 17.1 s | yes | 7.035 | 1003114961 |
| 2 | 16.9 s | yes | 7.208 | 67349777 |
| 3 | 17.2 s | yes | 7.405 | 1600027010 |
| 4 | 17.8 s | yes | 6.776 | 278416389 |
| 5 | 17.6 s | yes | 7.012 | 335779759 |
| 6 | 17.6 s | yes | 7.254 | 1600027010 |
| 7 | 17.2 s | yes | 6.96 | 982827070 |
| 8 | 17.6 s | yes | 6.973 | 567472345 |
| 9 | 17.9 s | yes | 6.63 | 567472345 |
| 10 | 17.6 s | yes | 6.691 | 1788449524 |
| 11 | 17.9 s | yes | 6.8 | 567472345 |
| 12 | 17.9 s | yes | 7.112 | 461348613 |
| 13 | 17.6 s | yes | 7.373 | 278416389 |
| 14 | 17.6 s | yes | 6.65 | 982827070 |
| 15 | 17.8 s | yes | 6.955 | 1600027010 |
| 16 | 17.9 s | yes | 7.015 | 1788449524 |
| 17 | 18.2 s | yes | 7.128 | 461348613 |
| 18 | 17.8 s | yes | 7.175 | 410500883 |
| 19 | 18.0 s | yes | 6.649 | 1003114961 |
| 20 | 17.9 s | yes | 6.868 | 461348613 |
| 21 | 17.8 s | yes | 6.998 | 461348613 |
| 22 | 17.4 s | yes | 7.084 | 410500883 |
| 23 | 18.0 s | yes | 6.834 | 232960483 |
| 24 | 18.0 s | yes | 7.252 | 410500883 |
| 25 | 17.9 s | yes | 7.115 | 1600027010 |

## No Solution Found

Every K in the run produced a saved `best_*` solution (or the run is incomplete).

## Results Summary

| K | Solution | max_ohm | best_score | Step | Seed |
|---|----------|---------|------------|------|------|
| 1 | yes | 7.035 | 7.035 | 1054 | 1003114961 |
| 10 | yes | 6.691 | 6.691 | 472 | 1788449524 |
| 11 | yes | 6.8 | 6.8 | 0 | 567472345 |
| 12 | yes | 7.112 | 7.112 | 3 | 461348613 |
| 13 | yes | 7.373 | 7.373 | 32 | 278416389 |
| 14 | yes | 6.65 | 6.65 | 1039 | 982827070 |
| 15 | yes | 6.955 | 6.955 | 150 | 1600027010 |
| 16 | yes | 7.015 | 7.015 | 696 | 1788449524 |
| 17 | yes | 7.128 | 7.128 | 0 | 461348613 |
| 18 | yes | 7.175 | 7.175 | 69 | 410500883 |
| 19 | yes | 6.649 | 6.649 | 286 | 1003114961 |
| 2 | yes | 7.208 | 7.208 | 990 | 67349777 |
| 20 | yes | 6.868 | 6.868 | 1 | 461348613 |
| 21 | yes | 6.998 | 6.998 | 0 | 461348613 |
| 22 | yes | 7.084 | 7.084 | 20 | 410500883 |
| 23 | yes | 6.834 | 6.834 | 595 | 232960483 |
| 24 | yes | 7.252 | 7.252 | 6 | 410500883 |
| 25 | yes | 7.115 | 7.115 | 4 | 1600027010 |
| 3 | yes | 7.405 | 7.405 | 48 | 1600027010 |
| 4 | yes | 6.776 | 6.776 | 165 | 278416389 |
| 5 | yes | 7.012 | 7.012 | 41 | 335779759 |
| 6 | yes | 7.254 | 7.254 | 1122 | 1600027010 |
| 7 | yes | 6.96 | 6.96 | 745 | 982827070 |
| 8 | yes | 6.973 | 6.973 | 0 | 567472345 |
| 9 | yes | 6.63 | 6.63 | 0 | 567472345 |

**K with saved solution:** **25/25**  


## Impedance comparison (generated vs real)

PEB: `latent_run.peb`  
Exported: `exported_samples/`  

### K = 01

![K01 impedance compare](exported_samples/K1/generated_vs_real_impedance_profile.png)

### K = 02

![K02 impedance compare](exported_samples/K2/generated_vs_real_impedance_profile.png)

### K = 03

![K03 impedance compare](exported_samples/K3/generated_vs_real_impedance_profile.png)

### K = 04

![K04 impedance compare](exported_samples/K4/generated_vs_real_impedance_profile.png)

### K = 05

![K05 impedance compare](exported_samples/K5/generated_vs_real_impedance_profile.png)

### K = 06

![K06 impedance compare](exported_samples/K6/generated_vs_real_impedance_profile.png)

### K = 07

![K07 impedance compare](exported_samples/K7/generated_vs_real_impedance_profile.png)

### K = 08

![K08 impedance compare](exported_samples/K8/generated_vs_real_impedance_profile.png)

### K = 09

![K09 impedance compare](exported_samples/K9/generated_vs_real_impedance_profile.png)

### K = 10

![K10 impedance compare](exported_samples/K10/generated_vs_real_impedance_profile.png)

### K = 11

![K11 impedance compare](exported_samples/K11/generated_vs_real_impedance_profile.png)

### K = 12

![K12 impedance compare](exported_samples/K12/generated_vs_real_impedance_profile.png)

### K = 13

![K13 impedance compare](exported_samples/K13/generated_vs_real_impedance_profile.png)

### K = 14

![K14 impedance compare](exported_samples/K14/generated_vs_real_impedance_profile.png)

### K = 15

![K15 impedance compare](exported_samples/K15/generated_vs_real_impedance_profile.png)

### K = 16

![K16 impedance compare](exported_samples/K16/generated_vs_real_impedance_profile.png)

### K = 17

![K17 impedance compare](exported_samples/K17/generated_vs_real_impedance_profile.png)

### K = 18

![K18 impedance compare](exported_samples/K18/generated_vs_real_impedance_profile.png)

### K = 19

![K19 impedance compare](exported_samples/K19/generated_vs_real_impedance_profile.png)

### K = 20

![K20 impedance compare](exported_samples/K20/generated_vs_real_impedance_profile.png)

### K = 21

![K21 impedance compare](exported_samples/K21/generated_vs_real_impedance_profile.png)

### K = 22

![K22 impedance compare](exported_samples/K22/generated_vs_real_impedance_profile.png)

### K = 23

![K23 impedance compare](exported_samples/K23/generated_vs_real_impedance_profile.png)

### K = 24

![K24 impedance compare](exported_samples/K24/generated_vs_real_impedance_profile.png)

### K = 25

![K25 impedance compare](exported_samples/K25/generated_vs_real_impedance_profile.png)
