---
title: exp038_true_multi
type: experiment
status: historical
era: vae
tags: [experiment, exp038_true_multi, historical, era-vae]
---

# exp038_true_multi

**Lineage:** [[exp037_lat_change]] → **exp038_true_multi** → [[exp039_improved_heatmap]]
**Status:** historical · **Era:** VAE era (17 Python files)

## Key hyperparameters

| Key | Value |
|-----|-------|
| `latent_dim` | `42` |
| `heatmap_private_dim` | `8` |
| `cond_dim` | `8` |
| `freq_fourier_features` | `8` |
| `num_epochs` | `925` |
| `batch_size` | `96` |
| `learning_rate` | `8e-06` |
| `train_split` | `0.9` |
| `split_by_design` | `True` |
| `layout_train_prob` | `0.4` |
| `heatmap_weight` | `2.75` |
| `impedance_weight` | `3.0` |
| `occupancy_weight` | `8.0` |
| `cross_freq_weight` | `1.0` |
| `data_dir` | `datasets/data_multifreq_norm` |
| `resume_checkpoint` | `experiments/exp038_true_multi/checkpoints/checkpoint_epoch_850.pt` |

*Full config: `experiments/exp038_true_multi/config.yaml` (120 keys)*

## Notes (from `experiments/exp038_true_multi/notes.md`)

# Experiment: exp038_true_multi

**Status:** Training completed (600 epochs) on **Quadro GV100** (fp16, no `torch.compile`).  
**Goal:** True multifreq PI heatmap VAE — continuous **PI_freq** on the heatmap branch only; shared **occupancy + impedance** per layout across seven anchor MHz.

---

## 1. Objective

| Branch | Conditioning | Role |
|--------|--------------|------|
| Heatmap | `K` + **PI_freq** (log10 norm, 1–600 MHz) | Frequency-specific PI spatial map |
| Occupancy | `K` only | Decap slot vector (52,) |
| Impedance | `K` only | Layout spectrum (231,) — **same curve at all MHz** for a design |

Compared to **exp035** (single-freq / random MHz per layout), exp038 uses **all seven MHz per layout** with design-group splits and frequency-balanced sampling.

---

## 2. Dataset

| Item | Value |
|------|--------|
| Raw | `C:\Users\muthusamy\Desktop\Raw` — flat layout: `heatmap_*MHz/`, `imp/`, `decap_combinations/all_combinations.csv` |
| Processed | `datasets/data_multifreq/` |
| Normalized | `datasets/data_multifreq_norm/` |
| Rows | **343,742** = **49,106 layouts × 7 MHz** |
| Anchors | **10, 63, 130, 200, 270, 400, 500 MHz** |
| Manifest | `datasets/data_multifreq/manifest.csv` — maps `sample_N` → `design_id`, `freq_mhz`, `pi_number` |
| Metadata cache | `datasets/data_multifreq_norm/multifreq_meta.json` — design hash (occ+imp) + freq bins |

**Processing notes**

- `all_combinations.csv` has **no header row** (49106 data lines); loader uses `header=None`.
- One row per **(layout, MHz)**; occ/imp duplicated 7× per layout (by design).

---

## 3. Model & code layout

```
experiments/exp038_true_multi/
  codes/
    train_vae_simple.py      # main training
    dataloader_multifreq.py  # design split + freq/K balancing
    vae_multi_input_simple.py
    physics_loss.py
    eval_val_recon.py
    inference_vae.py
    save_epoch1_losses.py    # backfill ep-1 into loss.csv
    refresh_exp038_plots.py  # dedupe CSV + rebuild plots
  checkpoints/               # checkpoint_epoch_{25,50,...,600}.pt, last_model.pt
  metrics/
    loss.csv                 # logged epochs (see §5)
    epoch_timing.csv         # per-epoch wall time
    timing.json
    plots/                   # VAETrainingLogger outputs (§7)
  logs/
    latent_stats.csv
  config.yaml                # snapshot at end of run
```

---

## 4. Training configuration (final run)

| Parameter | Value |
|-----------|--------|
| `num_epochs` | **600** |
| `batch_size` | **96** |
| `learning_rate` | **2e-5** → `ReduceLROnPlateau` (patience 25, factor 0.5, min 1e-6) |
| `train_samples_per_epoch` | **50,000** (random subset of ~309k train rows) |
| `train_split` | 0.9 by **design** (`split_by_design=True`) |
| `balance_freq` / `balance_k` | True |
| `val_on_checkpoint_only` | **True** (full val on ep 1, 25, 50, …, 600 only) |
| `checkpoint_interval` | **25** |
| `keep_last_n_checkpoints` | **0** (keep all interval checkpoints) |
| `cache_in_ram` | True (343k samples — long one-time load) |
| `amp` | **fp16** (GV100 — bf16/compile disabled) |
| `latent_dim` / `cond_dim` | 42 / 8 |

**Loss emphasis (exp035-style peaks)**

- `occupancy_weight=8`, `impedance_topk_k=20`, `impedance_topk_weight=7`, `impedance_under_penalty=2.8`
- Heatmap peak/grad/lap/contrast weights enabled; physics critic + anti-resonance on impedance

---

## 5. Metrics — `metrics/loss.csv`

Logged at **epochs 1, 25, 50, …, 600** (after dedupe; resume duplicates removed).  
Epoch **1** was backfilled with `save_epoch1_losses.py` (fresh init + seed 42, one train/val pass).

### 5.1 Summary table (checkpoint epochs)

| Epoch | Train total | Val total | Val heatmap | Val occ | Val imp | Val KL |
|------:|------------:|----------:|------------:|--------:|--------:|-------:|
| 1 | 11.28 | 5.46 | 0.642 | 0.336 | 0.147 | 0.773 |
| 25 | 8.03 | 5.31 | 0.198 | 0.305 | 0.546 | 1.330 |
| 100 | 6.20 | 3.79 | 0.115 | 0.238 | 0.282 | 1.735 |
| 200 | 5.13 | 2.95 | 0.097 | 0.175 | 0.197 | 1.780 |
| 300 | 4.51 | 2.47 | 0.093 | 0.134 | 0.161 | 1.794 |
| 400 | 4.14 | 2.13 | 0.086 | 0.103 | 0.140 | 1.815 |
| 500 | 3.78 | 1.86 | 0.085 | 0.078 | 0.126 | 1.812 |
| **600** | **3.60** | **1.64** | **0.080** | **0.060** | **0.116** | **1.860** |

### 5.2 Trends (600 vs 1)

| Metric | Epoch 1 (val) | Epoch 600 (val) | Change |
|--------|---------------|-----------------|--------|
| Total | 5.46 | **1.64** | −70% |
| Heatmap | 0.642 | **0.080** | −88% |
| Occupancy | 0.336 | **0.060** | −82% |
| Impedance | 0.147 | **0.116** | −21% |
| Physics RI | 0.016 | 0.006 | lower |
| Physics AR | 0.0026 | 0.0008 | lower |

Train total: **11.28 → 3.60**. Val recon improved steadily; **val KL rose** (~0.77 → 1.86) as β annealed (expected trade-off).

### 5.3 Physics (validation, epoch 600)

| Term | Value |
|------|------:|
| `val_physics_ri_loss` | 0.0059 |
| `val_physics_critic_sup_loss` | 0.0193 |
| `val_physics_ar_loss` | 0.00081 |

### 5.4 CSV maintenance

- **`loss.csv.bak`** — backup before dedupe (had duplicate rows per epoch from resume).
- Refresh / backfill: `codes/refresh_exp038_plots.py`, `codes/save_epoch1_losses.py`, `codes/metrics_csv_utils.py`.

---

## 6. Timing

From `metrics/timing.json` (600-epoch run):

| Metric | Value |
|--------|------:|
| Wall time | **21,601 s** (~**6.0 h**) |
| Avg train / epoch | **35.0 s** |
| Avg val (when run) | **23.5 s** |
| Est. avg epoch | **~36 s** (50k draws + skipped val on non-checkpoint epochs) |
| Full val runs | **25** (ep 1 + 24 checkpoints) |

`epoch_timing.csv` logs every epoch (train-only rows have `val_sec=0` when `val_on_checkpoint_only=True`). File may contain multiple lines per epoch if training was resumed; use last row per epoch for analysis.

**Rough planning**

| Epochs | ~Time |
|--------|------|
| 100 | ~1 h |
| 600 | ~6 h |

---

## 7. Plots (`metrics/plots/`)

Generated by `VAETrainingLogger` via:

```bash
python experiments/exp038_true_multi/codes/refresh_exp038_plots.py
```

### 7.1 Convergence (train losses)

![Convergence](metrics/plots/convergence_final.png)

Total / recon / KL and modality breakdown on **logged epochs** (1, 25, …, 600).

### 7.2 Loss components

![Loss components](metrics/plots/loss_components_final.png)

Heatmap, occupancy, and impedance train losses vs epoch.

### 7.3 Overfitting (train vs val)

![Overfitting](metrics/plots/overfitting_final.png)

Train–val gap for total, recon, and KL. Val total **5.46 → 1.64** with no sign of val blowing up at the end.

### 7.4 Physics losses

![Physics](metrics/plots/physics_losses_final.png)

Validation physics RI, critic supervision, and anti-resonance terms.

---

## 8. Latent statistics (`logs/latent_stats.csv`)

Snapshot at checkpoint epochs (validation pass):

| Epoch | β | Fused μ (approx) | Val recon | Val total |
|------:|--:|-----------------:|----------:|----------:|
| 1 | 0.00 | 0.06 | 5.35 | 5.46 |
| 25 | 0.01 | −0.02 | 5.16 | 5.31 |
| 300 | 0.08 | ~0 | 2.09 | 2.47 |
| 600 | 0.15 | 0.01 | **1.15** | **1.64** |

Fused posterior std ~**1.41** at epoch 600; expert KL on val ~**1.86**.

---

## 9. Checkpoints

| File | When |
|------|------|
| `checkpoints/checkpoint_epoch_{25,50,...,600}.pt` | Every 25 epochs |
| `checkpoints/last_model.pt` | End of training (epoch 600 weights) |

Resume: `resume_checkpoint="latest"` or path to specific `checkpoint_epoch_*.pt`.

---

## 10. How to run

### Training (fresh)

```bash
cd ~/gan && source /home/ubuntu/venv-cgan/bin/activate
python -u experiments/exp038_true_multi/codes/train_vae_simple.py
```

### Eval / inference

```bash
python experiments/exp038_true_multi/codes/eval_val_recon.py --ckpt last_model.pt
python experiments/exp038_true_multi/codes/inference_vae.py   # PI_FREQ_MHZ sweep
```

### Fix metrics & plots after the fact

```bash
# Optional: epoch-1 row if missing (GPU, ~1h without RAM cache)
python experiments/exp038_true_multi/codes/save_epoch1_losses.py

# Dedupe loss.csv + rebuild plots
python experiments/exp038_true_multi/codes/refresh_exp038_plots.py
```

---

## 11. Relation to exp035

| Topic | exp035 | exp038 |
|-------|--------|--------|
| Data | Often single-freq or row-wise split | **343k multifreq**, design split |
| PI_freq | Default 200 MHz if missing | **Per-row MHz** from `PI_freq/` |
| Dataloader | `create_data_loaders` | `create_multifreq_data_loaders` |
| Val @ ep 600 (approx) | ~4.59 total (prior run) | **~1.64** total |
| Latent opt | `Latent_opm/` uses exp035 ckpt | Use **exp038** `last_model.pt` for multifreq heatmaps |

---

## 12. Takeaways

1. **Multifreq training worked** — val heatmap/occ dropped sharply while impedance stayed controlled with top-k peak losses.
2. **50k samples/epoch** + **val on checkpoint only** made ~600 epochs feasible in ~6 h on GV100.
3. **Design-level split** is required for honest val with 7× row duplication per layout.
4. **loss.csv** is the source of truth for plots; always **dedupe** after resume before plotting.
5. For **downstream latent optimization**, impedance/occ branches still ignore PI_freq; heatmap/inference need explicit **MHz** at decode time.

---

## Phase 2 extension (epochs 601–750)

Resume from `checkpoint_epoch_600.pt` with **peak-aware impedance losses**.

**Why impedance “blew up” at epoch 601 (first broken run):** `impedance_loss` jumped ~0.29 → ~4.6 because (1) peak-index loss used **raw bin indices 0–230** with `log1p` → Huber terms of order 2–5, × weight 3; (2) new terms were applied at **full strength** with no ramp; (3) `reset_lr_on_resume` made the optimizer fight a new objective at full LR. Heatmap/occ kept improving because their losses were unchanged.

**Fix (re-run 601–750 from epoch-600 ckpt):** normalized peak index in [0,1], 40-epoch ramp (`impedance_peak_ramp_epochs`), lower dual/peak weights, `reset_lr_on_resume: false`, log `impedance_legacy_loss` vs `impedance_peak_loss`.

| Field | Value | Role |
|-------|------:|------|
| `impedance_peak_start_epoch` | 600 | Peak terms off for epochs &lt; 600 |
| `impedance_peak_ramp_epochs` | 40 | 0→1 scale over epochs 601–640 |
| `impedance_dual_topk_weight` | 1.0 | Dual top-k (recon + target) |
| `impedance_peak_index_weight` | 2.0 | Normalized peak **position** |
| `impedance_peak_mag_weight` | 1.5 | Magnitude at target peaks |

**Multifreq heatmap sweep (1–600 MHz, heatmap branch only):**

```bash
python scrap/generation/run_multifreq_heatmap_sweep.py --mode both
python experiments/exp038_true_multi/codes/eval_cross_freq.py
```

**PI-freq improvements (items 1–3):** Fourier+FiLM freq conditioning, cross-freq train loss,
`INFERENCE_MODE=layout` in sweep (occ/imp latent + per-MHz decode). Fine-tune from a prior ckpt
if loading old weights (`freq_conditioner` / FiLM layers are new).

```bash
python experiments/exp038_true_multi/codes/train_vae_simple.py
python experiments/exp038_true_multi/codes/surrogate_impedance.py
python experiments/exp038_true_multi/codes/eval_val_recon.py
```

Plot **`impedance_legacy_loss`** for apples-to-apples with pre-600 curves; total `impedance_loss` = legacy + peak.

## Phase 3 extension (epochs 751–850) — improve peak fit

Resume **`checkpoint_epoch_750.pt`** with:

| Change | Value |
|--------|--------|
| `learning_rate` | **8e-6** (fresh LR on resume) |
| `impedance_decoder_lr_mult` | **4×** on impedance encoder/decoder from ep 750 |
| Peak matching | **Greedy** nearest-neighbor (not sort-by-index) |
| `impedance_peak_index_weight` | 2.5 |
| `impedance_peak_mag_weight` | 2.0 |
| `impedance_dual_topk_weight` | 0.75 (less overlap with legacy top-k) |

```bash
python experiments/exp038_true_multi/codes/train_vae_simple.py
python experiments/exp038_true_multi/codes/surrogate_impedance.py
python experiments/exp038_true_multi/codes/eval_val_recon.py --ckpt last_model.pt
```

Track **`metrics/impedance_split.csv`** — target: val `impedance_peak` **&lt; 0.045**, legacy **&lt; 0.10**.

*Last updated from `loss.csv`, `timing.json`, `latent_stats.csv`, and `config.yaml` after 600-epoch completion and epoch-1 backfill.*

## Code modules

- [[experiments.exp038_true_multi.codes.compute_anchor_fg_max]] — Compute per-anchor foreground max (physical Ω) from data_multifreq_norm and write metrics JSON.
- [[experiments.exp038_true_multi.codes.dataloader_multifreq]] — Multi-frequency PI heatmap dataloader for exp038_true_multi.
- [[experiments.exp038_true_multi.codes.eval_cross_freq]] — Evaluate native vs cross-frequency heatmap reconstruction on val set.
- [[experiments.exp038_true_multi.codes.eval_val_recon]] — Validation reconstruction metrics for exp038.
- [[experiments.exp038_true_multi.codes.evaluate_vae]] — evaluate_vae.py — Post-training evaluation for exp025_latent_size_change
- [[experiments.exp038_true_multi.codes.freq_conditioning]] — PI frequency conditioning: Fourier features + MLP; FiLM for heatmap conv blocks.
- [[experiments.exp038_true_multi.codes.freq_inference_utils]] — PI frequency helpers for inference on unseen / off-anchor MHz (between training anchors).
- [[experiments.exp038_true_multi.codes.impedance_spectrum_loss]] — Peak-aware impedance losses (frequency weighting, dual top-k, peak alignment).
- [[experiments.exp038_true_multi.codes.inference_vae]] — Inference script for Multi-Input VAE — exp038_true_multi.
- [[experiments.exp038_true_multi.codes.metrics_csv_utils]] — Read/write exp038 metrics CSVs (dedupe by epoch, sorted).
- [[experiments.exp038_true_multi.codes.physics_loss]] — Physics-informed loss modules for exp031 Multi-Input VAE.
- [[experiments.exp038_true_multi.codes.refresh_exp038_plots]] — Update metrics/loss.csv (dedupe) and rebuild exp038 plots from VAETrainingLogger.
- [[experiments.exp038_true_multi.codes.save_epoch1_losses]] — Backfill epoch-1 train/val metrics into metrics/loss.csv (no checkpoint write).
- [[experiments.exp038_true_multi.codes.surrogate_impedance]] — Surrogate impedance model and training script — exp038_true_multi.
- [[experiments.exp038_true_multi.codes.train_vae_simple]] — Training — Multi-Input VAE (exp038_true_multi): multifreq PI heatmaps + peak losses.
- [[experiments.exp038_true_multi.codes.vae_multi_input_simple]]
- [[experiments.exp038_true_multi.codes.visualize_latent]] — visualize_latent.py — Latent space visualizations for exp025_latent_size_change

## Metrics artifacts

`experiments/exp038_true_multi/metrics/`

- `anchor_hm_fg_max.json`
- `epoch_timing.csv`
- `impedance_split.csv`
- `loss.csv`
- `loss.csv.bak`
- `off_anchor_eval_epoch_875.csv`
- `off_anchor_eval_epoch_900.csv`
- `off_anchor_eval_epoch_925.csv`
- `surrogate_train.log`
- `timing.json`

## Checkpoints

`checkpoint_epoch_100.pt`, `checkpoint_epoch_125.pt`, `checkpoint_epoch_150.pt`, `checkpoint_epoch_175.pt`, `checkpoint_epoch_200.pt`, `checkpoint_epoch_225.pt`, `checkpoint_epoch_25.pt`, `checkpoint_epoch_250.pt`, `checkpoint_epoch_275.pt`, `checkpoint_epoch_300.pt`, `checkpoint_epoch_325.pt`, `checkpoint_epoch_350.pt`, `checkpoint_epoch_375.pt`, `checkpoint_epoch_400.pt`, `checkpoint_epoch_425.pt`, `checkpoint_epoch_450.pt`, `checkpoint_epoch_475.pt`, `checkpoint_epoch_50.pt`, `checkpoint_epoch_500.pt`, `checkpoint_epoch_525.pt`, `checkpoint_epoch_550.pt`, `checkpoint_epoch_575.pt`, `checkpoint_epoch_600.pt`, `checkpoint_epoch_625.pt`, `checkpoint_epoch_650.pt`, `checkpoint_epoch_675.pt`, `checkpoint_epoch_700.pt`, `checkpoint_epoch_725.pt`, `checkpoint_epoch_75.pt`, `checkpoint_epoch_750.pt`, `checkpoint_epoch_775.pt`, `checkpoint_epoch_800.pt`, `checkpoint_epoch_825.pt`, `checkpoint_epoch_850.pt`, `checkpoint_epoch_875.pt`, `checkpoint_epoch_900.pt`, `checkpoint_epoch_925.pt`, `last_model.pt`, `surrogate_best.pt`, `surrogate_epoch_100.pt`, `surrogate_epoch_150.pt`, `surrogate_epoch_200.pt`, `surrogate_epoch_50.pt`, `surrogate_last.pt`
