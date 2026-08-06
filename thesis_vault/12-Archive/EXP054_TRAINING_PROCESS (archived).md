---
title: EXP054_TRAINING_PROCESS (archived)
type: archive
source: docs/_archive/EXP054_TRAINING_PROCESS.md
tags: [archive]
---

> [!caution] Archived implementation note — not for thesis citation.

# exp054: Full training process

Everything performed during training from `run_train_gpu1.sh` through teardown.

---

## At a glance

```
run_train_gpu1.sh
  └─ train_vae_simple.main()
       └─ train_vae()  [exp054 patches]
            └─ train_core.train_vae()  [vendored training loop]
                 ├─ setup (config, dataloader, model, resume)
                 └─ for epoch in 1..400:
                      ├─ on_train_epoch_start (hm breakdown, schedule log)
                      ├─ train epoch (~187 batches × forward/loss/backward)
                      ├─ val epoch (ep 1, every 25, final)
                      ├─ LR scheduler step (when val ran)
                      ├─ log CSVs (checkpoint epochs)
                      └─ checkpoint + off-anchor eval (every 25 / 50)
                 └─ last_model.pt + plots + timing summary
```

---

## 0. Shell launch (`run_train_gpu1.sh`)

- `cd` to repo root
- Pin GPU (`CUDA_VISIBLE_DEVICES=1`), thread env vars
- Start **nohup** `python -m experiments.exp054_K_30.codes.train_vae_simple`
- Log → `logs/exp054_K30_fresh_<timestamp>.log`
- PID → `logs/exp054_K30_train.pid`

---

## 1. Module entry (`main()`)

- Set `VAE_EXPERIMENT_DIR` → `experiments/exp054_K_30`
- Set `VAE_DATA_DIR` from `config.yaml` if present
- Call `train_vae()`

---

## 2. exp054 bootstrap (`train_vae()` in exp054)

**`_patch_training()`** monkey-patches the vendored `train_core` trainer:

| Patch | Effect |
|--------|--------|
| `Config` | exp054 config fields |
| `build_vae_model` | `MultiInputVAEPoeFreq` (PoE + layout private head) |
| `_run_epoch` | exp054 `run_epoch_encode._run_epoch` |
| `heatmap_loss` | Pearson + grad + peak/valley loc |
| `vae_loss` | exp054 loss stack (+ log1p peak/valley) |
| `_on_stats_loaded` | unbounded norm bundle, force fp32 |
| `create_multifreq_data_loaders` | exp054 dataloader |
| `_build_optimizer` | separate impedance LR from ep 80 |
| `_append_impedance_split_csv` | + `heatmap_peak_split.csv` |
| `_apply_yaml_config` | hooks `on_train_epoch_start` + `eval_off_anchor.set_off_anchor_config` |

Off-anchor eval is wired in `train_core` via `eval_off_anchor.py` (interval gating → single `off_anchor_eval.csv`).

Then: load yaml → print training summary → call `train_core.train_vae()`.

---

## 3. One-time setup (`train_core.train_vae()`)

### Config & dirs

- Load `config.yaml`, map `*_frac` → epoch boundaries (`apply_curriculum_epochs`)
- Adapt AMP/TF32 for GPU
- Resolve resume checkpoint
- **Fresh run:** delete old `checkpoints/`, `logs/`, `metrics/`
- Create `checkpoints/`, `logs/`, `metrics/`, `metrics/plots/`

### Data & stats

- Read `normalization_stats.json` → clip bounds, `background_value`, impedance log-std
- **exp054:** load unbounded `NormStatsBundle`, set `physics_fg_clip_min`, force `amp=off`

### DataLoader (`exp054 dataloader_multifreq`)

- Validate multifreq dataset
- `VAEDataset` (layout store required, optional RAM cache)
- **Split by design** (90/10 train/val)
- Train subset: `_IndexedSubset` with **cross-freq alt-MHz pairs** (`heatmap_norm_alt`, `PI_freq_alt`)
- **30,000 random train draws/epoch** (replacement), shuffle (no K/freq balance)
- Val loader: standard collate, no cross-freq pairs

### Model

- Build **MultiInputVAEPoeFreq** (UNet skips, occ spatial tower, layout private head, freq PoE expert)
- Optional `torch.compile` (off in current config)
- **PhysicsLoss** module (RI + critic + AR) if physics weights > 0

### Resume (if checkpoint)

- Load weights, optimizer, optionally scheduler
- Restore/clamp curriculum epoch boundaries
- Reset LR on resume (`reset_lr_on_resume: true`)

### Optimizer

- AdamW on all trainable params
- From **ep 80+:** split groups — backbone LR vs impedance LR × `impedance_decoder_lr_mult` (0.5)
- **ReduceLROnPlateau** on val loss

---

## 4. Every epoch (×400)

### 4a. Epoch start hooks

**`on_train_epoch_start`** (exp054):

- Print prior-epoch heatmap breakdown (tier_a, peak/valley log1p)
- Every 25 epochs: log schedule weights (hm / imp / cf / layout_prob)

### 4b. Schedule scalars (recomputed each epoch)

| Scalar | What it does |
|--------|----------------|
| `beta` | KL anneal 0 → 0.1 → 0.11 (done ~ep 240) |
| `modality_dropout` | Anneal 0.06 over first 32 epochs; 0.03 in heatmap-focus phase |
| `_physics_weights` | Physics RI/AR warmup ep 0–60 |
| `_penalty_scale` | Impedance penalty warmup ep 0–20 |
| `_phase_weights` | After ep 260: hm_w=3.5, imp_w=1.5, cf_w=1.0 (heatmap focus) |
| `modality_dropout_protect_heatmap` | On from ep 260 — don't drop heatmap modality |

### 4c. Training pass (`_run_epoch`, train=True)

**Per batch** (~30k / 160 ≈ 187 batches, `drop_last=True`):

1. **Load batch** — heatmap, occ, imp, K, PI_freq (+ alt pair for cross-freq)
2. **`_prepare_batch`** — H2D, shape fix, heatmap z-clip, FG mask for encode input
3. **Forward** (`_forward_train_batch`, full fp32):
   - Encode heatmap path → `z_post`, U-Net skips
   - Optional layout latent for distillation
   - **~55% layout path** / **~45% encode path** (`layout_train_prob`)
   - Layout path: decode from layout-z; optional output distill vs teacher decode
   - Encode path: decode with teacher skips
   - **PI_freq jitter** on decode (60% prob, log10 σ=0.1)
4. **Loss** (`_vae_loss_exp054`):
   - **Heatmap tier_a:** Pearson + grad vector/dir + peak_loc + valley_loc
   - **Occupancy:** focal BCE (+ K-based pos_weight in base; not K-reweighted in exp054)
   - **Impedance:** spectrum + deriv + topk + peaks (peak terms ramp ep 80–119)
   - **KL** + free-bits + μ hinge/bias + per-expert KL + σ regularization
   - **Physics** (if active): RI, critic sup, AR on reconstructions
   - **Log1p peak:** hotspot + max + centroid (×3.5)
   - **Log1p valley:** coldspot + min + centroid (×2.0); ×1.75 extra in focus phase
   - No K/MHz sample weighting (`apply_k=False`)
5. **Extra train-only terms:**
   - Latent distill (layout μ → encode μ), weight 2.0
   - Output distill (layout decode → teacher decode), weight 2.5
   - **Cross-modal** (every 4 batches): encode occ/imp-only → decode all modalities
   - **Cross-freq** (from ep 20): decode at alt MHz vs alt heatmap; reuse encode-z + skips
   - KAN spline L1 (if enabled)
6. **Backward** — grad clip 1.0, skip non-finite batches (`training_skip_nonfinite_batches`)
7. **Accumulate** loss metrics on GPU

### 4d. Validation (when scheduled)

**Runs on:** epoch 1, every **25** epochs, final epoch (`val_on_checkpoint_only: true`)

Same `_run_epoch` with `train=False`:

- Standard encode→decode (AMP if enabled for val)
- All loss terms (no distill / cross-freq / cross-modal / backward)
- Collect latent stats (global + per-modality + per-K if saving checkpoint)

### 4e. LR scheduler

- `ReduceLROnPlateau.step(val_loss)` only when val actually ran

### 4f. Logging (checkpoint epochs + ep 1)

- Append **`metrics/loss.csv`** (train + val core losses)
- Append **`metrics/impedance_split.csv`** (legacy vs peak impedance)
- Append **`metrics/heatmap_peak_split.csv`** (tier_a, peak/valley log1p)
- Append **`logs/latent_stats.csv`**
- Append **`metrics/epoch_timing.csv`** (train/val seconds)
- Print epoch summary line to stdout

### 4g. Checkpoint epochs (every 25)

- Save **`checkpoints/checkpoint_epoch_{N}.pt`** (model, opt, sched, config, latent stats)
- Update **`metrics/timing.json`**
- **Off-anchor eval** (every **50** epochs only):
  - 3 MHz: 100, 270, 400
  - 12 val batches × encode_cross + layout_cross
  - Append rows to **`metrics/off_anchor_eval.csv`** (MSE, Pearson, peak_loc)

---

## 5. End of training (after 400 epochs)

- Write final **`metrics/timing.json`** summary
- Save **`checkpoints/last_model.pt`**
- Plots → `metrics/plots/`:
  - `convergence_final.png`
  - `loss_components_final.png`
  - `overfitting_final.png`
  - `physics_losses_final.png` (if physics on)
- Print logger statistics
- Write final **`experiments/exp054_K_30/config.yaml`** snapshot

---

## 6. Removed / not running

These were removed in earlier exp054 cleanup and are **not** part of the current pipeline:

- Append-tag sampling curriculum
- High-MHz pair bias / synthetic freq blend
- Layout-path epoch curriculum (fixed 0.55 / 0.45)
- K/freq balanced sampling
- Decoder freezing
- Per-epoch off-anchor CSV files (now single `off_anchor_eval.csv`)

---

## Schedule quick reference (400 epochs)

| Epoch | Event |
|------:|-------|
| 20 | Cross-freq loss on |
| 80 | Impedance peak ramp + separate impedance LR |
| 240 | β anneal done |
| 260 | Heatmap focus phase + protect heatmap in dropout |
| 25, 50, 75… | Checkpoint + val |
| 50, 100, 150… | Off-anchor eval appended to CSV |

---

## Key config knobs (current)

| Setting | Value |
|---------|-------|
| `num_epochs` | 400 |
| `checkpoint_interval` | 25 |
| `train_samples_per_epoch` | 30000 |
| `batch_size` | 160 |
| `layout_train_prob` | 0.55 |
| `cross_freq_layout_mix_prob` | 0.45 |
| `eval_off_anchor_mhz` | 100, 270, 400 |
| `eval_off_anchor_max_batches` | 12 |
| `eval_off_anchor_interval` | 50 |
| `balance_k` / `balance_freq` | false |
| Decoder freeze | none |

---

## Output file index

| Path | When written |
|------|----------------|
| `logs/exp054_K30_fresh_*.log` | Shell launch (repo `logs/`) |
| `metrics/loss.csv` | Checkpoint epochs |
| `metrics/impedance_split.csv` | Checkpoint epochs |
| `metrics/heatmap_peak_split.csv` | Checkpoint epochs |
| `logs/latent_stats.csv` | Checkpoint epochs |
| `metrics/epoch_timing.csv` | Every epoch |
| `metrics/timing.json` | Each checkpoint + end |
| `metrics/off_anchor_eval.csv` | Every 50 epochs at checkpoint |
| `checkpoints/checkpoint_epoch_*.pt` | Every 25 epochs |
| `checkpoints/last_model.pt` | End of training |
| `metrics/plots/*.png` | End of training |

---

## Related docs

- EXP054_SELF_CONTAINED.md (`EXP054_SELF_CONTAINED.md`) — vendored modules (no exp038 imports)
- EXP054_TRAINING_SCHEDULE_RETUNE.md (`EXP054_TRAINING_SCHEDULE_RETUNE.md`) — schedule retune rationale
- EXP054_OFF_ANCHOR_EVAL_SINGLE_FILE.md (`EXP054_OFF_ANCHOR_EVAL_SINGLE_FILE.md`) — off-anchor eval format
