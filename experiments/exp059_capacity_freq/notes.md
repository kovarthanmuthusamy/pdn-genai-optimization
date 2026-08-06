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
