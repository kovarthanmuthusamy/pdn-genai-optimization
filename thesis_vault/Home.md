---
title: Home
type: moc
tags: [moc]
---

# PDN Generative-Surrogate Thesis Vault

Knowledge graph generated from the `genai_pdn` codebase by `tools/build_vault.py`.
Re-run that script after code changes — it rewrites `10-Concepts`, `12-Archive`,
`20-Experiments`, `30-Code`, `40-Datasets`, `50-Results` and leaves `00-Thesis/`
and `15-Ideas/` (your own writing) untouched.

## Start here

- [[Thesis Outline]] — chapter map, what to write where
- [[Claim Ledger]] — every claim with its evidence artifact
- [[Reading Paths]] — guided tours through the graph
- [[Literature Review]] — curated papers, mapped to chapters
- [[Experiment Lineage]] — the full arc, GAN era through to the current model
- [[framework-overview]] — the framework in one page

## Current model track

- [[exp059_capacity_freq]] — **current**
- [[exp060_multitype_occ]] — **exploratory**

## Concepts

- [[active-learning]]
- [[cursor-handoff]]
- [[data-pipeline]]
- [[dataset]]
- [[docs_update_ahk_removal]]
- [[ecadstar_ahk_winactivate_fix]]
- [[ecadstar_headless_cli]]
- [[ecadstar_persistence_and_locking]]
- [[evaluation-metrics]]
- [[evaluation-suite]]
- [[experiment-lineage]]
- [[framework-overview]]
- [[gnn-rationale]]
- [[gp-error-surrogate]]
- [[impedance-surrogate]]
- [[latent-optimization]]
- [[limitations-and-validity]]
- [[model-architecture]]
- [[multitype_peb_pipeline]]
- [[normalization-and-losses]]
- [[training-procedure]]

## Experiments

[[exp001_new_architecture]], [[exp002_hyper_parms_change]], [[exp003]], [[exp004]], [[exp005]], [[exp007]], [[exp008]], [[exp009]], [[exp010]], [[exp011]], [[exp029_heat_private]], [[exp037_lat_change]], [[exp038_true_multi]], [[exp039_improved_heatmap]], [[exp040]], [[exp041]], [[exp042]], [[exp043]], [[exp044]], [[exp045]], [[exp046]], [[exp047]], [[exp048]], [[exp049]], [[exp050]], [[exp051_new_datas_appended]], [[exp052_unbounded_pearson]], [[exp053_peak_log1p_losses]], [[exp054_K_30]], [[exp055_hard_occ]], [[exp056_graph_vae]], [[exp057_structured_graph]], [[exp058_asymmetric_kl]], [[exp059_capacity_freq]], [[exp060_multitype_occ]], [[exp_simple]]

## Results

- [[acquisition_ab_validation]]
- [[acquisition_ab_validation_exp059]]
- [[al_exp057_001]]
- [[al_exp057_gp_error_smoke]]
- [[al_exp058_001]]
- [[al_exp059_gp_error_001]]
- [[hole_finding_exp059]]

## Datasets

- [[data_multifreq_al_overlay_exp057]]
- [[data_multifreq_al_overlay_exp058]]
- [[data_multifreq_al_overlay_exp059]]
- [[data_multifreq_al_overlay_exp060]]
- [[data_multifreq_gmax]]
- [[data_multifreq_norm]]
- [[data_multifreq_norm_z_score]]
- [[data_multifreq_train]]
- [[data_multifreq_train_norm_robust]]
- [[data_multifreq_train_norm_unbounded]]

## Code by area

| Area | Modules | Index |
|------|---------|-------|
| `experiments/` | 431 | [[experiments (code index)]] |
| `pipelines/` | 79 | [[pipelines (code index)]] |
| `scrap/` | 30 | [[scrap (code index)]] |
| `active_learning_pi/` | 26 | [[active_learning_pi (code index)]] |
| `scratch/` | 21 | [[scratch (code index)]] |
| `tools/` | 17 | [[tools (code index)]] |
| `src_vae/` | 16 | [[src_vae (code index)]] |
| `libs/` | 10 | [[libs (code index)]] |
| `evaluation/` | 8 | [[evaluation (code index)]] |
| `_root/` | 4 | [[_root (code index)]] |
| `Rules/` | 3 | [[Rules (code index)]] |
| `viewer/` | 2 | [[viewer (code index)]] |

## Archive

- [[Archive Index]] — historical implementation notes (not citable)

**Totals:** 647 modules · 22 concepts · 36 experiments · 7 AL runs · 10 datasets
