# Experiment: exp042

PI-frequency PoE expert on the **exp041 inverse-K dataset** — same training stack as
exp041/exp039, with MHz entering the latent on the **layout path** via a dedicated
Product-of-Experts (PoE) expert on heatmap-private dimensions.

---

## 1. Goal

Improve **off-anchor MHz generalization** for PI-Distribution heatmaps when inference
uses the **layout path** (no GT heatmap at encode time):

```text
z = encode_layout(occ, imp, K, PI_ref)   →   decode(z, K, PI_target)
```

exp041 only conditioned MHz at **decode** (FiLM). exp042 adds **PI_freq into z** on
encode via PoE, aligned with multifreq PEB sweep and `cross_freq` training.

---

## 2. Dataset (shared with exp041)

exp042 trains on the **same subsampled multifreq dataset** built for exp041.

### 2.1 Path

```text
datasets/data_multifreq_norm
```

(`data_dir` in `config.yaml`; absolute path on ki-gpu:
`/home/ubuntu/gan/datasets/data_multifreq_norm`)

### 2.2 Why subsample?

The full multifreq corpus used ~**1000 layouts per K** for every decap count. At
**high K**, many placements produce similar PI responses (saturated / redundant).
At **low K**, layouts are more diverse and more informative for learning spatial
PI structure.

**exp041** applied **inverse-exponential K weighting** at **design_id** level: entire
layouts (and all their MHz rows) are kept or deleted together.

### 2.3 Subsample policy

Script: `datasets/subsample_multifreq_inverse_k.py`

```text
n_keep(K) = clamp(n_min, n_max, round(n_ref * exp(-(K - k_anchor) / τ_eff)))

τ_eff = tau_low   when K ≤ k_low_max   (default K≤20, τ_low=35 — gentle decay)
τ_eff = tau       when K > k_low_max   (default τ=12 — sharper drop at high K)
```

| Parameter | Default | Role |
|-----------|---------|------|
| `n_ref` | 1000 | Reference count at `k_anchor` |
| `k_anchor` | 2 | K where exp factor = 1 |
| `tau_low` | 35 | Low/mid K region — keep more diversity |
| `tau` | 12 | High K — stronger downsample |
| `k_low_max` | 20 | Boundary between τ regimes |
| `n_min` | 150 | Floor layouts per K |
| `n_max` | 1000 | Cap layouts per K |
| `seed` | 42 | Reproducible layout selection |

Edge K `{0, 1, 51, 52}` are dropped by default (`DEFAULT_EDGE_K`).

### 2.4 Scale after subsample

| Quantity | Approx. value |
|----------|----------------|
| Manifest rows | **~194,985** |
| Unique layouts (`design_id`) | **~19,499** |
| Rows per layout | 10 anchor MHz (multifreq bins) |

Each row: occupancy layout + impedance + heatmap @ one training anchor frequency.

### 2.5 On-disk layout

```text
data_multifreq_norm/
  manifest.csv              # sample_name, design_id, freq_mhz, …
  layouts/{design_id}/      # shared occupancy per layout
  heatmap/{stem}.npy        # z-score heatmap per (layout, MHz)
  PI_freq/{stem}.npy        # normalized PI frequency
  normalization_stats.json
```

Training uses `split_by_design: true` (no layout leakage train/val).

### 2.6 Dataloader balancing (train time)

Even after subsampling, the trainer still applies light rebalance (exp041/exp042):

| Knob | Value | Notes |
|------|-------|-------|
| `k_balance_power` | **0.35** | Lighter than exp039 — dataset already K-weighted |
| `balance_freq` | true | `freq_balance_power: 1.2` |
| `train_samples_per_epoch` | **35000** | ~90% of ~175k train rows sampled per epoch |
| `stratify_by_k` | true | |

---

## 3. Architecture (exp042 vs exp041)

Base: `MultiInputVAE` from `experiments/exp038_true_multi/codes/vae_multi_input_simple.py`.

exp042 model: `MultiInputVAEPoeFreq` in `codes/vae_poe_freq.py`.

### 3.1 Latent split

| Component | Dims | Role |
|-----------|------|------|
| Shared | 34 | occ + imp PoE experts |
| Private (heatmap) | 8 | heatmap expert + **freq PoE expert** |
| **Total** | **42** | `latent_dim` |

### 3.2 PoE experts

```text
Full encode (training with GT heatmap):
  z = PoE( heatmap_expert, occ_expert, imp_expert, freq_expert(PI_freq), prior )

Layout encode (sweep / layout_train / cf loss):
  z = PoE( occ_expert, imp_expert, freq_expert(PI_freq), prior )
```

| Expert | Conditions | Writes to |
|--------|------------|-----------|
| Occupancy | K only | shared 34 |
| Impedance | K only | shared 34 |
| Heatmap | K + PI_freq (FiLM) | full 42 (when GT hm present) |
| **Freq PoE** | **PI_freq only** | **private 8** |
| Prior | — | full 42 |

Freq expert: `freq_conditioner(PI_freq) → Linear → μ, logσ` on private dims (+144 params
vs exp041).

**Decode unchanged:** `z + K + PI_freq` FiLM on heatmap decoder.

### 3.3 Why private 8 dims?

Matches existing `heatmap_private_dim` split: occ/imp stay layout+K; MHz-specific
variation is routed to the subspace that already carried heatmap-private information.
(User has discussed **single z** / `heatmap_private_dim: 0` for a future exp043 if
decap-combo coupling remains an issue.)

---

## 4. Sweep-aligned training

All three mechanisms needed for ECAD multifreq sweep are implemented.

| Mechanism | Config | Code path |
|-----------|--------|-----------|
| Layout encode path | `layout_train_prob: 0.92` | `_forward_train_batch` → `encode_layout_latent` |
| Cross-freq `cf` loss | `cross_freq_weight: 0.5`, `cross_freq_layout_z_only: true` | `_cross_freq_heatmap_loss` |
| Freq PoE in layout z | `use_freq_poe_expert: true` | `vae_poe_freq.encode_layout_latent` |

**Encode/decode MHz convention (matches inference `mode="layout"`):**

- **Encode:** native / `pi_ref` MHz in z (freq PoE + occ/imp)
- **Decode:** target MHz via FiLM

`cf` loss: `z = encode_layout(occ, imp, K, pi_native)` → `decode(z, K, pi_alt)` vs GT
heatmap at alt frequency.

---

## 5. Training config summary

File: `config.yaml` (same curriculum as exp041 unless noted).

| Phase | Epochs | Highlights |
|-------|--------|------------|
| Warmup | 1–20 | `cross_freq` starts @ ep 20 |
| Main | 20–200 | `cross_freq_weight: 0.5`, `layout_train_prob: 0.92` |
| Heatmap focus | 200–400 | `heatmap_weight: 5.0`, `cross_freq: 0.75`, lower imp weight |

Key knobs:

```yaml
num_epochs: 400
learning_rate: 5e-5          # → 2.5e-5 after LR schedule in late training
layout_train_prob: 0.92
cross_freq_layout_z_only: true
heatmap_focus_start_epoch: 200
use_freq_poe_expert: true
synthetic_blend_prob: 0.08   # exp042 train script only
```

Trainer: `experiments/exp042/codes/train_vae_simple.py`  
(shared loop: `experiments/exp038_true_multi/codes/train_vae_simple.py`)

---

## 6. Training results (400 epochs, completed)

**Runtime:** ~932 min (~15.5 h) on ki-gpu  
**Best val loss:** **7.89** (epoch 400)  
**Checkpoint:** `checkpoints/checkpoint_epoch_400.pt`

### 6.1 Final train losses (epoch 400)

| Loss | Final | Run min |
|------|-------|---------|
| total | 12.56 | 10.87 |
| recon | 9.58 | 8.36 |
| heatmap | **1.47** | **1.47** |
| occupancy | 0.20 | 0.20 |
| impedance | 0.47 | 0.43 |
| kl | 1.75 | 0.43 |

`cf` (cross-freq) ~**1.95** at end — still active in heatmap-focus phase.

### 6.2 Off-anchor eval (`layout_cross` MSE, val set)

Diagnostic: layout z → decode @ off-anchor MHz vs native-row GT heatmap.  
Eval MHz: **100, 175, 350** (between training anchors).

| Epoch | 100 MHz | 175 MHz | 350 MHz |
|-------|---------|---------|---------|
| 100 | 8.5 | 17.6 | 26.8 |
| 150 | **13.1** | 34.0 | 50.0 |
| 200 | 17.0 | 41.6 | 57.1 |
| 250 | 19.6 | 28.3 | 41.3 |
| 300 | 23.6 | 32.8 | 48.0 |
| **400** | **26.4** | **38.8** | **53.7** |

**Best offline layout_cross:** roughly epochs **75–150** (lowest @ 100 MHz).  
After heatmap-focus (ep 200+), on-anchor heatmap improved but **layout_cross crept up**
(same pattern as exp041 late training).

### 6.3 Comparison vs exp041 @ epoch 400

| MHz | exp042 layout_cross | exp041 layout_cross |
|-----|---------------------|---------------------|
| 100 | **26.4** | 229 |
| 175 | **38.8** | 342 |
| 350 | **53.7** | 376 |

Freq PoE gives **~6–9×** lower layout_cross MSE vs exp041 at the same epoch count.
exp041’s best layout_cross was earlier (e.g. ep 200: 92 / 141 / 142) — exp042 is
still much better at ep 400 than exp041 at ep 400.

---

## 7. Inference and CAD sweep

### 7.1 Inference modes

| Mode | Use |
|------|-----|
| `layout` | PEB sweep: encode @ `pi_ref`, decode @ target MHz |
| `anchor_blend` | Off-anchor with anchor statistics blend |
| `marginal` | Sample z from prior / marginal |

Sweep pipeline: `scrap/run_multifreq_sweep_pipeline.py`  
(generate → ECADStar batch → move/compare/report)

### 7.2 Recommended checkpoints for sweep

| Checkpoint | When to use |
|------------|-------------|
| `checkpoint_epoch_400.pt` | Full training; best val + on-anchor heatmap |
| `checkpoint_epoch_150.pt` or `200.pt` | Best **offline** layout_cross if sweep looks blurry off-anchor |

Update `CHECKPOINT_PATH` in pipeline / `run_multifreq_heatmap_sweep.py`.

### 7.3 ECAD workflow

1. `python scrap/run_multifreq_sweep_pipeline.py` (config in script header)
2. Or step-by-step: generate → ECADStar → `scrap/multifreq_move_and_compare.py`
3. Reports copied to `C:\Users\muthusamy\Desktop\reports`

---

## 8. Code map

```text
experiments/exp042/
  config.yaml
  notes.md
  codes/
    vae_poe_freq.py          # MultiInputVAEPoeFreq
    train_vae_simple.py      # Config + model hook + synthetic blend
    inference_vae.py
    evaluate_vae.py
    exp042_eval_common.py
  checkpoints/
    checkpoint_epoch_*.pt
    last_model.pt
  metrics/
    off_anchor_eval_epoch_*.csv
    epoch_timing.csv
    plots/
```

---

## 9. Relation to other experiments

| Exp | Dataset | Model change | Status |
|-----|---------|--------------|--------|
| exp039 | Full multifreq (~1000/K) | Baseline multifreq VAE | Reference |
| **exp041** | **Inverse-K subsample** | Same as exp039 | 475 ep; layout_cross degrades late |
| **exp042** | **Same as exp041** | **+ freq PoE on private 8** | **400 ep complete** |
| exp043 (idea) | exp041 | Single z, freq PoE on full latent | Not built |

---

## 10. Commands

```bash
# Train (fresh)
python experiments/exp042/codes/train_vae_simple.py

# Resume (edit config.yaml: resume_checkpoint, num_epochs, LR/weights)
python experiments/exp042/codes/train_vae_simple.py

# Offline eval
python experiments/exp042/codes/evaluate_vae.py

# Off-anchor diagnostic only
python experiments/exp038_true_multi/codes/eval_cross_freq.py \
  --ckpt experiments/exp042/checkpoints/checkpoint_epoch_400.pt

# Full CAD sweep pipeline
python scrap/run_multifreq_sweep_pipeline.py
```

---

## 11. Resume / improvement notes

Late training (ep 200→400) trades **layout_cross** for lower on-anchor heatmap loss.

If resuming for marginal gains:

- Resume from **400** (+50–75 ep) with **higher** `heatmap_focus_cross_freq_weight` (~1.0)
  and **lower** `heatmap_focus_heatmap_weight` (~4.0)
- Monitor `layout_cross` every 25 epochs; stop if it rises 2 checks in a row
- Alternatively evaluate **`checkpoint_epoch_150`** or **`200`** for sweep without
  more training

---

## 12. Open questions

1. Does offline layout_cross improvement transfer to ECAD `.map` quality at 10/70/120/270/400 MHz?
2. Is **8 private dims** enough for freq + decap interaction, or is **single z** (exp043) better?
3. Should heatmap-focus phase be shortened or cross-freq weight raised to avoid ep-200+ regression?
