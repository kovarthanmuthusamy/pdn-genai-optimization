---
title: exp059_capacity_freq
type: experiment
status: current
era: vae
tags: [experiment, exp059_capacity_freq, current, era-vae]
---

# exp059_capacity_freq

**Lineage:** [[exp058_asymmetric_kl]] → **exp059_capacity_freq** → [[exp060_multitype_occ]]
**Status:** current · **Era:** VAE era (21 Python files)

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
| `al_overlay_data_dir` | `datasets/data_multifreq_al_overlay_exp059` |
| `resume_checkpoint` | `experiments/exp059_capacity_freq/checkpoints/last_model.pt` |

*Full config: `experiments/exp059_capacity_freq/config.yaml` (183 keys)*

## Config variants

- `config_al_finetune.runtime.yaml`
- `config_al_finetune.yaml`
- `config_occ_only_finetune.yaml`

## Notes (from `experiments/exp059_capacity_freq/notes.md`)

# exp059_capacity_freq

## Goal
Fix **mid-frequency heatmap averaging** observed vs ECAD: low/high anchor MHz
are sharp, mid-band (~150–280 MHz, higher spatial complexity) collapses to a
blurry conditional mean. Root causes traced in exp057/exp058:
1. Frequency conditioning too shallow/narrow (`cond_dim=8`, FiLM at 1 decoder scale).
2. Small spatial-private latent (`heatmap_private_dim=15`) shared across 24 MHz.
3. MSE-family losses return the mean → blur on complex mid targets.

## What changed vs exp058
- **Full-encode path only** (AL disabled): `layout_train_prob=0`, `occ_only_encode_prob=0`,
  `latent_distill_weight=0`, `output_distill_weight=0`, `eval_use_occ_only_layout=false`,
  `al_overlay_data_dir=null`, `al_finetune_early_stop=false`. AL comes later.
- **High latent capacity**: `latent_dim 65→128`, `heatmap_private_dim 15→40` (shared=80).
- **Wider + deeper freq conditioning**: `cond_dim 8→32`, `freq_fourier_features 8→16`,
  `use_multiscale_film=true` → FiLM at every decoder scale (8×8, 16×16, 32×32, 64×64)
  instead of a single 32-ch injection. This is the key fix so mid-band frequencies can
  carve *distinct* spatial structure.
- **Frequency-weighted spectral loss** (anti-blur, no GAN instability):
  2D-FFT log-magnitude L1 between recon and target, with a Gaussian MHz bump
  (`heatmap_spectral_mid_*`) up-weighting the mid band. Keys: `heatmap_spectral_weight=1.5`,
  `heatmap_spectral_mid_boost=1.5`, `center=200 MHz`, `width=80 MHz`.
- Cross-freq kept ON (uses full-encode z) — helps frequency generalization.
- Fresh train: `resume_checkpoint=null`, `learning_rate=2e-4`, `num_epochs=500`.

## Adversarial (GAN) loss — decision
Deliberately **NOT enabled**. A GAN as a *replacement* for MSE hallucinates
plausible-but-wrong detail and breaks pixel accuracy needed for downstream peak
detection. Correct usage is reconstruction-anchor + small adversarial add-on
(pix2pix/SRGAN). Plan: first measure how far capacity + multi-scale FiLM +
spectral loss get mid-band sharpness; only then add a PatchGAN discriminator
**conditioned on PI_freq** as a small secondary term. Kept out for now to isolate
variables and avoid instability ("implement only what's needed").

## How to train (fresh, full-encode)
```bash
cd /home/ubuntu/genai_pdn && source venv
export CUDA_VISIBLE_DEVICES=0
export VAE_EXPERIMENT_DIR=$(pwd)/experiments/exp059_capacity_freq
unset VAE_CONFIG_PATH
python -m experiments.exp059_capacity_freq.codes.train_vae_simple
```

## What to watch
- Per-MHz off-anchor eval at 155/250/265 (now full-encode): mid FG-MSE down, Pearson up.
- `heatmap_spectral_loss` in metrics should fall alongside sharper mid maps.


## Update
- U-Net skip connections **removed from the model code** (OptionalSkipFuse / `_last_heatmap_skips` / `heatmap_skips` args gone). Decoder is latent+FiLM only — matches stage-2.


## Layout holdout split (2026-07-20) — **retrain required**
`_split_indices_by_design` now holds out **whole layouts** (all MHz of a
`design_id` go to train XOR val). Previous code split frequencies *within*
each design (same layout in both splits) — bad for GP residual / AL eval.

- Deterministic: `seed=42`, `train_split=0.9` → ~22481 train / ~2498 val layouts.
- `resume_checkpoint=null` for a fresh run; prior artifacts archived under
  `archive_pre_layout_holdout_*`.
- No other training/loss changes required for this split.

### Foreground train (single GPU)
```bash
cd /home/ubuntu/genai_pdn
source /home/ubuntu/venv-cgan/bin/activate
export CUDA_VISIBLE_DEVICES=0
export VAE_EXPERIMENT_DIR=$(pwd)/experiments/exp059_capacity_freq
unset VAE_CONFIG_PATH
python -m experiments.exp059_capacity_freq.codes.train_vae_simple
```

### Foreground train (2-GPU DDP)
```bash
cd /home/ubuntu/genai_pdn
source /home/ubuntu/venv-cgan/bin/activate
export CUDA_VISIBLE_DEVICES=0,1
export VAE_EXPERIMENT_DIR=$(pwd)/experiments/exp059_capacity_freq
unset VAE_CONFIG_PATH
torchrun --standalone --nproc_per_node=2 -m experiments.exp059_capacity_freq.codes.train_vae_simple
```

## Code modules

- [[experiments.exp059_capacity_freq.codes.__init__]]
- [[experiments.exp059_capacity_freq.codes.dataloader_base]] — Multi-frequency PI heatmap dataloader (vendored for exp055).
- [[experiments.exp059_capacity_freq.codes.dataloader_multifreq]] — exp055 multifreq dataloader — cross-freq pairs + optional K/freq balance.
- [[experiments.exp059_capacity_freq.codes.distributed_train]] — DDP helpers for exp055 multi-GPU training.
- [[experiments.exp059_capacity_freq.codes.eval_off_anchor]] — Off-anchor eval hook for exp055 training checkpoints.
- [[experiments.exp059_capacity_freq.codes.eval_spatial_metrics]] — Off-anchor eval — spatial metrics, append rows to one CSV (epoch column).
- [[exp059_common]] — exp059 shared paths, yaml config, and VAE constructor kwargs.
- [[experiments.exp059_capacity_freq.codes.graph_imp]] — 1D spectrum GNN for PI impedance (231 bins) — exp059.
- [[experiments.exp059_capacity_freq.codes.graph_occ]] — Graph message-passing occupancy encoder/decoder for exp056 Graph VAE.
- [[experiments.exp059_capacity_freq.codes.heatmap_peak_losses]] — Heatmap losses: Pearson+grad tier_a + log1p peak/valley extrema.
- [[experiments.exp059_capacity_freq.codes.impedance_spectrum_loss]] — Impedance spectrum loss for exp055 — lean stack, no redundant terms.
- [[experiments.exp059_capacity_freq.codes.inference_vae]] — Inference script for Multi-Input VAE — exp059 (no U-Net skips).
- [[experiments.exp059_capacity_freq.codes.occupancy_binary]] — Binary occupancy for heatmap decode — matches CAD / ECADStar discrete layouts.
- [[experiments.exp059_capacity_freq.codes.physics_loss]] — Physics-informed loss modules for exp031 Multi-Input VAE.
- [[experiments.exp059_capacity_freq.codes.run_epoch_encode]] — exp055 training epoch — layout path + log1p peak/valley.
- [[experiments.exp059_capacity_freq.codes.spatial_metrics]] — FG Pearson + soft extrema location metrics.
- [[experiments.exp059_capacity_freq.codes.train_core]] — exp055 training core loop (self-contained).
- [[experiments.exp059_capacity_freq.codes.train_vae_simple]] — Train VAE for exp059 Graph VAE — tier_a + peak/valley extrema losses.
- [[experiments.exp059_capacity_freq.codes.training_guard]] — Finite-loss / NaN guards for exp055 training.
- [[experiments.exp059_capacity_freq.codes.vae_multi_input_simple]]
- [[experiments.exp059_capacity_freq.codes.vae_poe_freq]] — Multi-input VAE — freq PoE private dims + encode-first (no U-Net skips).

## Metrics artifacts

`experiments/exp059_capacity_freq/metrics/`

- `epoch_timing.csv`
- `heatmap_peak_split.csv`
- `impedance_split.csv`
- `loss.csv`
- `off_anchor_eval.csv`
- `timing.json`

## Checkpoints

`best_off_anchor_model.pt`, `checkpoint_epoch_1.pt`, `checkpoint_epoch_10.pt`, `checkpoint_epoch_100.pt`, `checkpoint_epoch_105.pt`, `checkpoint_epoch_110.pt`, `checkpoint_epoch_115.pt`, `checkpoint_epoch_120.pt`, `checkpoint_epoch_125.pt`, `checkpoint_epoch_130.pt`, `checkpoint_epoch_135.pt`, `checkpoint_epoch_140.pt`, `checkpoint_epoch_145.pt`, `checkpoint_epoch_15.pt`, `checkpoint_epoch_150.pt`, `checkpoint_epoch_155.pt`, `checkpoint_epoch_160.pt`, `checkpoint_epoch_165.pt`, `checkpoint_epoch_170.pt`, `checkpoint_epoch_175.pt`, `checkpoint_epoch_180.pt`, `checkpoint_epoch_185.pt`, `checkpoint_epoch_190.pt`, `checkpoint_epoch_195.pt`, `checkpoint_epoch_20.pt`, `checkpoint_epoch_200.pt`, `checkpoint_epoch_205.pt`, `checkpoint_epoch_210.pt`, `checkpoint_epoch_215.pt`, `checkpoint_epoch_220.pt`, `checkpoint_epoch_225.pt`, `checkpoint_epoch_230.pt`, `checkpoint_epoch_235.pt`, `checkpoint_epoch_240.pt`, `checkpoint_epoch_245.pt`, `checkpoint_epoch_25.pt`, `checkpoint_epoch_250.pt`, `checkpoint_epoch_255.pt`, `checkpoint_epoch_260.pt`, `checkpoint_epoch_265.pt`, `checkpoint_epoch_270.pt`, `checkpoint_epoch_275.pt`, `checkpoint_epoch_280.pt`, `checkpoint_epoch_285.pt`, `checkpoint_epoch_290.pt`, `checkpoint_epoch_295.pt`, `checkpoint_epoch_30.pt`, `checkpoint_epoch_300.pt`, `checkpoint_epoch_305.pt`, `checkpoint_epoch_310.pt`, `checkpoint_epoch_315.pt`, `checkpoint_epoch_320.pt`, `checkpoint_epoch_325.pt`, `checkpoint_epoch_330.pt`, `checkpoint_epoch_335.pt`, `checkpoint_epoch_340.pt`, `checkpoint_epoch_345.pt`, `checkpoint_epoch_35.pt`, `checkpoint_epoch_350.pt`, `checkpoint_epoch_355.pt`, `checkpoint_epoch_360.pt`, `checkpoint_epoch_365.pt`, `checkpoint_epoch_370.pt`, `checkpoint_epoch_375.pt`, `checkpoint_epoch_380.pt`, `checkpoint_epoch_385.pt`, `checkpoint_epoch_390.pt`, `checkpoint_epoch_395.pt`, `checkpoint_epoch_40.pt`, `checkpoint_epoch_400.pt`, `checkpoint_epoch_405.pt`, `checkpoint_epoch_410.pt`, `checkpoint_epoch_415.pt`, `checkpoint_epoch_420.pt`, `checkpoint_epoch_425.pt`, `checkpoint_epoch_430.pt`, `checkpoint_epoch_435.pt`, `checkpoint_epoch_440.pt`, `checkpoint_epoch_445.pt`, `checkpoint_epoch_45.pt`, `checkpoint_epoch_450.pt`, `checkpoint_epoch_455.pt`, `checkpoint_epoch_460.pt`, `checkpoint_epoch_465.pt`, `checkpoint_epoch_470.pt`, `checkpoint_epoch_475.pt`, `checkpoint_epoch_480.pt`, `checkpoint_epoch_485.pt`, `checkpoint_epoch_5.pt`, `checkpoint_epoch_50.pt`, `checkpoint_epoch_55.pt`, `checkpoint_epoch_60.pt`, `checkpoint_epoch_65.pt`, `checkpoint_epoch_70.pt`, `checkpoint_epoch_75.pt`, `checkpoint_epoch_80.pt`, `checkpoint_epoch_85.pt`, `checkpoint_epoch_90.pt`, `checkpoint_epoch_95.pt`, `last_model.pt`
