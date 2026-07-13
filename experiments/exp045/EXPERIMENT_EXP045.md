# Experiment exp045 — Unbounded z-score + spatial-pattern training

Builds on exp044 with **Option B** normalization and targeted fixes for high-Ω peaks, cross-freq stability, and **spatial heatmap quality**.

## What changed vs exp044

| Area | exp044 | exp045 |
|---|---|---|
| **Dataset** | `data_multifreq_norm_z_score` (99.9% clip → ~34 Ω) | **`data_multifreq_unbounded`** (full log-z range → ~84 Ω) |
| **Recon clip** | `heatmap_clip_recon: true` | **`false`** |
| **Heatmap loss** | Huber δ=1, uniform fg | **Huber δ=2 + tail weight** (z > 3) |
| **Phys p99 loss** | hard z-clip before exp | **soft cap** at z=6 |
| **Architecture** | flat heatmap encoder | **U-Net skips** (exp043) |
| **Occ spatial** | 52-d vector only | **7×8 conv tower → fuse @ 16/32/64** |
| **Spatial losses** | grad/lap only | **+ Pearson fg + peak-location** |
| **Native encode decode** | z only | **+ teacher U-Net skips** |
| **Eval** | fg-MSE only | **+ Pearson r + peak_loc_err per MHz** |
| **Early stop** | — | **encode_cross @ 330 MHz** (best ckpt saved) |
| **Freq private dims** | 8 | **12** |
| **latent_dim** | 42 | **48** (shared=36) |
| **layout_train_prob** | 0.15 | **0.30** |
| **Cross-freq** | weight 1.0, focus @ ep200 | **weight 1.5**, focus delayed **@ ep400** |
| **β final** | 0.11 | **0.09** (less KL pressure) |
| **Cross-freq decode** | no U-Net skips | **teacher skips** on encode path |

## Spatial-pattern improvements (all 4 implemented)

### 1. Losses — Pearson + peak location

```yaml
heatmap_pearson_weight: 1.0      # 1 - Pearson r on foreground
heatmap_peak_loc_weight: 0.5     # soft-argmax peak coordinate L2
heatmap_grad_weight: 2.5
heatmap_lap_weight: 3.0
```

Implemented in `codes/unbounded_heatmap_loss.py` + `codes/spatial_metrics.py`.

### 2. Training — teacher skips on native encode batches

When `layout_train_prob` does **not** fire (~70% of batches), decode uses:

```text
z = encode(GT hm)     skips = heatmap_skips from encoder
decode(z, π, occ, heatmap_skips=skips)
```

Implemented in `codes/run_epoch_encode.py` (`encode_native_teacher_skips: true`).

### 3. Architecture — occ spatial tower (multi-scale)

```text
52-d occ → 7×8 grid → conv tower → features @ 16×16, 32×32, 64×64
                                    → fused into heatmap decoder at each scale
```

`use_occ_spatial_tower: true`, `occ_spatial_ch: 8` in `vae_multi_input_simple.py`.

### 4. Eval + early stop

At each checkpoint, `off_anchor_eval_epoch_*.csv` includes:

| Column | Meaning |
|---|---|
| `hm_fg_mse_mean` | Foreground MSE |
| `pearson_fg_mean` | Spatial correlation (higher = better) |
| `peak_loc_err_mean` | Normalized peak distance (lower = better) |

- **Best checkpoint**: `checkpoints/best_encode_cross_spatial.pt` when `encode_cross @ 330 MHz` MSE improves
- **Early stop**: after `early_stop_encode_cross_patience` (default 3) checkpoints without MSE improvement at 330 MHz

## Build unbounded dataset

```bash
cd /home/ubuntu/gan
bash pipelines/normalize/build_unbounded.sh
# → datasets/data_multifreq_unbounded/
```

## Train

```bash
.venv/bin/python experiments/exp045/codes/train_vae_simple.py
```

## Why layout sweep barely changed (exp044 finding)

**Occupancy + impedance do not uniquely determine heatmap spatial pattern.**  
Layout z only sees which pads are on and the impedance curve — not where current concentrates on the board.

The occ spatial tower helps the **layout decode path** by injecting pad locations at multiple scales; it does not fully replace GT heatmap spatial information for marginal sweep.

### Fair sweep eval (recommended)

```text
INFERENCE_MODE = "layout"
# pass dataset occ/imp from the same design_id (oracle layout)
```

Or compare **encode path** with GT heatmap for latent-opt validation.

## Evaluation protocol

| Metric | Use for |
|---|---|
| `val heatmap_loss` | Native π encode quality |
| `encode_cross @ 330/400 MHz` MSE | Cross-freq magnitude guardrail |
| `pearson_fg_mean @ 330 MHz` | Spatial pattern quality |
| `peak_loc_err_mean` | Hotspot placement |
| `best_encode_cross_spatial.pt` | Best spatial checkpoint (not last) |
| Oracle layout sweep | Upper bound on sweep without GT hm |
| Marginal sweep | Lower bound (generative prior only) |

## Code map

```text
experiments/exp045/
  config.yaml
  codes/
    train_vae_simple.py         # patches + early-stop hook
    unbounded_heatmap_loss.py   # robust tail-weighted + spatial losses
    spatial_metrics.py          # Pearson, peak-loc helpers
    eval_spatial_metrics.py     # off-anchor eval with spatial cols
    run_epoch_encode.py         # teacher skips on encode + cross-freq
    vae_multi_input_simple.py   # U-Net skips + occ spatial tower
    vae_poe_freq.py
pipelines/normalize/
  multifreq.py                  # USE_UNBOUNDED_HEATMAP flag
  build_unbounded.sh
```

## Joint latent optimization

```bash
.venv/bin/python experiments/exp045/codes/latent_optimization_joint.py \
  --checkpoint experiments/exp045/checkpoints/best_encode_cross_spatial.pt \
  --sample-index 0 --mhz 330 --steps 300
```

Physical Ω for plots: `exp(z * log_std + log_mean) - 1` (no clip when unbounded).
