# Experiment 046 — Bounded Data + Spatial Pattern Losses

## v3 — Latent Distillation for layout→heatmap generation (current)

### Why
The train-vs-val diagnostic (`metrics/diag_train_split.csv` vs `diag_val_split.csv`, on `checkpoint_epoch_300.pt`) showed the real bottleneck:

| Path | Train r | Val r |
|------|---------|-------|
| encode_cross @ 330 | 0.903 | 0.913 |
| **layout_cross @ 330** | **0.812** | **0.739** |

The **encode** path (autoencode the real heatmap) is excellent everywhere. The **layout** path (generate from occupancy+impedance+freq, no heatmap) is weak **even on training designs**. So the limiter is not dataset sparsity but the **learnability of `f(occ, imp, freq) → heatmap`**, and the fact that the **heatmap-private latent dims + U-Net skips are empty during generation** — exactly the info the decoder learned to rely on.

### The split latent
```
latent_dim 65 = shared_latent_dim 50 + heatmap_private_dim 15
```
- **encode path:** private 20 dims filled by the heatmap expert (real spatial structure) + freq PoE expert.
- **layout path (before v3):** private 20 dims filled by the **freq expert only** → carries frequency, not *where the hotspots are* for this layout.

### v3 change: latent distillation (teacher → student)
Make the layout path **predict the heatmap's contribution to the private dims**, then train it to match the encode path.

1. **`layout_private_head`** (new module in the model) — MLP `[occ_feat(64) + imp_feat(64) + cond(K+freq)] → heatmap_private_dim`. Added as an extra PoE expert in `encode_layout_latent_full`, playing the role the heatmap expert plays during encode. The layout latent now fills all 65 dims.
2. **Distillation loss** (`run_epoch_encode._latent_distill_loss`):
   ```
   L_distill = ‖ mu_layout[shared] − sg(mu_encode[shared]) ‖²
             + private_mult · ‖ mu_layout[private] − sg(mu_encode[private]) ‖²
   ```
   `sg` = stop-grad (teacher frozen). Private dims weighted `latent_distill_private_mult×` (split-latent core: align the spatial-structure dims hardest).
3. **`layout_train_prob` 0.15 → 0.30** — decode from the (now richer) layout latent on more batches, training the full generation pipeline.

At inference, `encode_layout_latent` automatically routes through the private head — no heatmap needed, but the latent is now close to the teacher's, so the already-good decoder produces sharper hotspots.

### v3 config additions
| Key | Value |
|-----|-------|
| `use_layout_private_head` | true |
| `latent_distill_weight` | 1.0 |
| `latent_distill_private_mult` | 3.0 |
| `layout_train_prob` | 0.30 |

### Files touched (v3)
- `codes/vae_multi_input_simple.py` — `layout_private_mu/logvar` modules + flag; `_encode_impedance_expert(return_feat=...)`
- `codes/vae_poe_freq.py` — `encode_layout_latent_full()`; `encode_layout_latent` routes through it
- `codes/run_epoch_encode.py` — `_latent_distill_loss`, distill term in train loop + accumulation
- `codes/train_vae_simple.py` — Config fields, `build_vae_model`, summary print
- `codes/exp046_eval_common.py` — `load_model` passes `use_layout_private_head`
- `config.yaml` — v3 keys above

### Diagnostic tool
`codes/eval_train_vs_val.py` — runs the spatial eval on both train and val splits for a checkpoint (writes `metrics/diag_train_split.csv` / `diag_val_split.csv`). Re-run after training to measure the layout_cross train/val gap closing.

---

# Experiment 046 — Bounded Data + Spatial Pattern Losses (v2 — plateau fixes)

## Goal
Test whether spatial-pattern improvements (Pearson correlation loss, peak location loss, occupancy spatial tower, U-Net teacher skips) actually work **on bounded data** where exp044 already showed good convergence (val hm_loss ~1.17).

**Rationale:** exp045 introduced both unbounded data AND spatial losses simultaneously. The unbounded data scale caused gradient conflicts and plateau. exp046 isolates the spatial changes on the known-good bounded foundation to determine if they help or hurt.

## v2 Changes (plateau fix — fresh start)

The first run of exp046 plateaued at epoch ~225 (val heatmap loss ~1.00, spatial metrics stalled). Root-cause analysis identified three issues:

1. **Unused s8 skip** — The encoder produced 128ch 8×8 features (`s8`) but the decoder never fused them. This was the strongest bottleneck-level spatial signal, wasted.
2. **Insufficient heatmap private dims** — 12 private dims couldn't carry enough spatial information through the latent bottleneck.
3. **Loss competition** — Impedance peak loss starting at epoch 200 created gradient conflicts with heatmap spatial learning.

### v2 fixes applied:

| Change | Before | After |
|--------|--------|-------|
| **s8 skip fusion** | Not connected | `OptionalSkipFuse(128,128,128)` before `dec_up1` |
| **heatmap_private_dim** | 12 | **15** |
| **freeze_occ_imp_epoch** | N/A | **300** (freeze occ/imp decoders to focus on heatmap) |
| **heatmap_focus_impedance_weight** | 2.5 | **1.0** (reduce gradient competition) |
| **impedance_peak_start_epoch** | 200 | **300** (delay impedance peak to avoid early conflict) |
| **impedance_peak_focus_epoch** | 200 | **350** |
| **early_stop_min_epoch** | 150 | **200** |
| **early_stop_patience** | 4 | **5** |

## Key Differences from exp044

| Feature | exp044 | exp046 v2 |
|---------|--------|-----------|
| Dataset | `data_multi_norm` (bounded log1p z-score) | **Same** |
| Heatmap loss | Standard MSE + penalty terms | + **Pearson correlation** (w=0.5) + **Peak location** (w=0.3) |
| Architecture | Base MultiInputVAE + PoE + U-Net skips | + **OccSpatialTower** + **s8/s16/s32 skips** |
| Teacher skips | No | **Yes** — encoder skips passed to decoder on encode batches |
| HM private dim | 8 | **15** |
| Freeze occ/imp | N/A | Epoch 300+ |
| Impedance peak | epoch 200 | **epoch 300** (delayed) |
| Early stopping | None | encode_cross@330MHz, patience=5, min_epoch=200 |

## Spatial Losses (added to standard bounded heatmap loss)

1. **Pearson Foreground Correlation** (`heatmap_pearson_weight=0.5`)
   - `1 - pearson_r(recon_fg, target_fg)` — penalizes shape differences, not just magnitude

2. **Peak Location Loss** (`heatmap_peak_loc_weight=0.3`)
   - Soft-argmax peak coordinate distance (normalized L2) — penalizes hotspot displacement
   - Applied only on full loss (not lite/cross-freq)

## Architecture

- **Raw z-score output** — no sigmoid; decoder outputs unbounded values matching bounded z-score targets [-1.69, 4.07]
- **Three U-Net skips**: s32 (16ch), s16 (32ch), s8 (128ch) — all fused in decoder
- **s8 skip fusion** — `OptionalSkipFuse(128,128,128)` fuses the 8×8 bottleneck-level encoder features with the FC-reshaped latent before upsampling
- **OccSpatialTower** — conv tower on 7×8 occupancy → multi-scale features (16, 32, 64) fused into decoder
- **Teacher skips** — encoder heatmap features passed directly to decoder during native encode batches (85% of training)
- **Increased capacity** — latent_dim 65, heatmap_private_dim 15

## Training Curriculum

| Epoch | Event |
|-------|-------|
| 0 | Training starts |
| 20 | Cross-freq loss activated |
| 200 | Heatmap focus phase (hm_w=5.0, imp_w=1.0) + early-stop monitoring begins |
| 300 | Impedance peak loss starts + occ/imp decoders frozen |
| 350 | Impedance peak focus |
| 600 | Max epochs (or early stop) |

## Early Stopping

- Monitors `encode_cross` FG MSE at 330 MHz
- Patience: 5 checkpoint evaluations without improvement
- **min_epoch=200**: no early stop before epoch 200
- Saves `best_encode_cross_spatial.pt` on each improvement

## Run

```bash
cd ~/gan
source ~/venv-cgan/bin/activate
python experiments/exp046/codes/train_vae_simple.py
```
