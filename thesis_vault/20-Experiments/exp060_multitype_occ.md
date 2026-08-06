---
title: exp060_multitype_occ
type: experiment
status: exploratory
era: vae
tags: [experiment, exp060_multitype_occ, exploratory, era-vae]
---

# exp060_multitype_occ

**Lineage:** [[exp059_capacity_freq]] → **exp060_multitype_occ** → [[exp_simple]]
**Status:** exploratory · **Era:** VAE era (22 Python files)

## Key hyperparameters

| Key | Value |
|-----|-------|
| `latent_dim` | `128` |
| `heatmap_private_dim` | `40` |
| `cond_dim` | `32` |
| `freq_fourier_features` | `16` |
| `num_epochs` | `510` |
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
| `al_overlay_data_dir` | `datasets/data_multifreq_al_overlay_exp060` |
| `resume_checkpoint` | `experiments/exp060_multitype_occ/checkpoints/last_model.pt` |

*Full config: `experiments/exp060_multitype_occ/config.yaml` (184 keys)*

## Config variants

- `config_al_finetune.yaml`
- `config_occ_only_finetune.yaml`

## Notes (from `experiments/exp060_multitype_occ/notes.md`)

# exp060 — Multi-type occupancy (one-hot types, no empty channel)

Fork of **exp059_capacity_freq**. Occupancy schema:

\[
\mathrm{occ} \in \{0,1\}^{52 \times T}
\]

- **Empty** = all-zero row `[0, 0, …]`
- **Type t** = one-hot on channel `t-1`
- Config: `n_decap_types` (= \(T\)); model `n_occ_classes = T`
- Default: **T = 2** → shape `(52, 2)`

## Examples (T=2)

| State | Vector |
|-------|--------|
| empty | `[0, 0]` |
| type 1 | `[1, 0]` |
| type 2 | `[0, 1]` |

## Loss / decode

- Decoder logits `(B, 52, T)` + **sigmoid / focal BCE** (not softmax CE — empty is not a class)
- Hard CAD: top-K by max type score → argmax type; others zero
- Heatmap spatial cond uses **presence** = sum of type channels

## Status

- Legacy binary `occ.npy` maps to type-1 only (channel 0).
- exp059 checkpoints are not compatible.

## Code modules

- [[experiments.exp060_multitype_occ.codes.__init__]]
- [[experiments.exp060_multitype_occ.codes.dataloader_base]] — Multi-frequency PI heatmap dataloader (vendored for exp055).
- [[experiments.exp060_multitype_occ.codes.dataloader_multifreq]] — exp055 multifreq dataloader — cross-freq pairs + optional K/freq balance.
- [[experiments.exp060_multitype_occ.codes.distributed_train]] — DDP helpers for exp055 multi-GPU training.
- [[experiments.exp060_multitype_occ.codes.eval_off_anchor]] — Off-anchor eval hook for exp055 training checkpoints.
- [[experiments.exp060_multitype_occ.codes.eval_spatial_metrics]] — Off-anchor eval — spatial metrics, append rows to one CSV (epoch column).
- [[exp060_common]] — exp060 shared paths, yaml config, and VAE constructor kwargs.
- [[experiments.exp060_multitype_occ.codes.graph_imp]] — 1D spectrum GNN for PI impedance (231 bins) — exp060.
- [[experiments.exp060_multitype_occ.codes.graph_occ]] — Graph message-passing occupancy encoder/decoder for multi-type occ (exp060).
- [[experiments.exp060_multitype_occ.codes.heatmap_peak_losses]] — Heatmap losses: Pearson+grad tier_a + log1p peak/valley extrema.
- [[experiments.exp060_multitype_occ.codes.impedance_spectrum_loss]] — Impedance spectrum loss for exp055 — lean stack, no redundant terms.
- [[experiments.exp060_multitype_occ.codes.inference_vae]] — Inference script for Multi-Input VAE — exp060 (no U-Net skips).
- [[experiments.exp060_multitype_occ.codes.occupancy_binary]] — Occupancy harden helpers (legacy binary + multi-type one-hot).
- [[occupancy_types]] — Multi-type decap occupancy: one-hot over catalog types only (no empty channel).
- [[experiments.exp060_multitype_occ.codes.physics_loss]] — Physics-informed loss modules for exp031 Multi-Input VAE.
- [[experiments.exp060_multitype_occ.codes.run_epoch_encode]] — exp055 training epoch — layout path + log1p peak/valley.
- [[experiments.exp060_multitype_occ.codes.spatial_metrics]] — FG Pearson + soft extrema location metrics.
- [[experiments.exp060_multitype_occ.codes.train_core]] — exp055 training core loop (self-contained).
- [[experiments.exp060_multitype_occ.codes.train_vae_simple]] — Train VAE for exp060 Graph VAE — tier_a + peak/valley extrema losses.
- [[experiments.exp060_multitype_occ.codes.training_guard]] — Finite-loss / NaN guards for exp055 training.
- [[experiments.exp060_multitype_occ.codes.vae_multi_input_simple]]
- [[experiments.exp060_multitype_occ.codes.vae_poe_freq]] — Multi-input VAE — freq PoE private dims + encode-first (no U-Net skips).

## Checkpoints

*none on disk (untracked)*
