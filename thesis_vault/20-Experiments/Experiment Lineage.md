---
title: Experiment Lineage
type: moc
tags: [moc, experiment, lineage]
---

# Experiment Lineage

36 experiment directories under `experiments/`, ordered by numeric prefix.
Era is detected from the source: GAN markers (discriminator, adversarial, WGAN) versus
VAE markers (reparameterisation, logvar, KL, posterior).

**Era split:** 2 GAN · 26 VAE · 5 undetermined · 3 empty

> [!note] No GAN → VAE transition detected yet.
> Either the early experiments are not copied in, or their code carries no GAN
> markers. Re-run `tools/build_vault.py` after copying exp001–exp036.

## The arc

| # | Experiment | Era | Files | Headline |
|---|-----------|-----|-------|----------|
| 1 | [[exp001_new_architecture]] | GAN | 0 | Try new architecture of GAN for the PDN optimization |
| 2 | [[exp002_hyper_parms_change]] | ? | 0 | Critic was very ocsillative |
| 3 | [[exp003]] | ? | 0 | increasing laten dimension to(256)  see the results and aslo added reconstruction losses |
| 4 | [[exp004]] | ? | 0 |  |
| 5 | [[exp005]] | ? | 0 | trying out new method for Occupancy_map |
| 6 | [[exp007]] | GAN | 0 | removing MAE reconstruction losses instead using mid layers of the discriminator for comparing the features in |
| 7 | [[exp008]] | ? | 0 | try equally sized dataset |
| 8 | [[exp009]] | VAE | 0 | try out VAE model instead of GAN |
| 9 | [[exp010]] | VAE | 1 | trying out SSIM instead of MSE loss |
| 10 | [[exp011]] | — | 0 |  |
| 11 | [[exp029_heat_private]] | — | 0 |  |
| 12 | [[exp037_lat_change]] | VAE | 7 |  |
| 13 | [[exp038_true_multi]] | VAE | 17 | **Status:** Training completed (600 epochs) on **Quadro GV100** (fp16, no `torch.compile`). |
| 14 | [[exp039_improved_heatmap]] | VAE | 26 |  |
| 15 | [[exp040]] | VAE | 29 | See [docs/implementation.md](docs/implementation.md) for architecture, config, and commands. |
| 16 | [[exp041]] | VAE | 26 | Train the exp039-style multifreq PI VAE on the **inverse-K subsampled** dataset |
| 17 | [[exp042]] | VAE | 9 | PI-frequency PoE expert on the **exp041 inverse-K dataset** — same training stack as |
| 18 | [[exp043]] | VAE | 18 |  |
| 19 | [[exp044]] | VAE | 11 |  |
| 20 | [[exp045]] | VAE | 15 |  |
| 21 | [[exp046]] | VAE | 12 |  |
| 22 | [[exp047]] | VAE | 14 |  |
| 23 | [[exp048]] | VAE | 14 |  |
| 24 | [[exp049]] | VAE | 14 |  |
| 25 | [[exp050]] | VAE | 15 |  |
| 26 | [[exp051_new_datas_appended]] | VAE | 17 | Fine-tune the exp050 Tier-A VAE on the expanded multifreq training set |
| 27 | [[exp052_unbounded_pearson]] | VAE | 19 | Fork of exp051 with: |
| 28 | [[exp053_peak_log1p_losses]] | VAE | 19 | **Mode:** resume from `checkpoint_epoch_325.pt`, run ep326→450 with stronger log1p peak stack. |
| 29 | [[exp054_K_30]] | VAE | 19 | 400-epoch heatmap + impedance training on unified multifreq dataset (K≈30 focus via experiment design, not los |
| 30 | [[exp055_hard_occ]] | VAE | 20 | exp054 successor: **binary occupancy before heatmap decode** so inference/QC matches CAD discrete layouts. |
| 31 | [[exp056_graph_vae]] | VAE | 21 | exp054 successor: **binary occupancy before heatmap decode** so inference/QC matches CAD discrete layouts. |
| 32 | [[exp057_structured_graph]] | VAE | 22 | exp054 successor: **binary occupancy before heatmap decode** so inference/QC matches CAD discrete layouts. |
| 33 | [[exp058_asymmetric_kl]] | VAE | 22 | **Fresh train from scratch** (does **not** load exp057 weights) with: |
| 34 | [[exp059_capacity_freq]] | VAE | 21 | Fix **mid-frequency heatmap averaging** observed vs ECAD: low/high anchor MHz |
| 35 | [[exp060_multitype_occ]] | VAE | 22 | Fork of **exp059_capacity_freq**. Occupancy schema: |
| 36 | [[exp_simple]] | — | 0 |  |

## Related

- [[experiment-lineage]] — the curated thesis narrative in `docs/`
- [[Thesis Outline]] — where the arc is argued
