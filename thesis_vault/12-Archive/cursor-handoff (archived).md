---
title: cursor-handoff (archived)
type: archive
source: docs/_archive/cursor-handoff.md
tags: [archive]
---

> [!caution] Archived implementation note — not for thesis citation.

# Cursor handoff — VAE multifreq heatmap project

Last updated: 2026-06-23. Use `@docs/cursor-handoff.md` at the start of new chats.

---

## 1. Workspace setup


| Item                 | Value                                                                                   |
| -------------------- | --------------------------------------------------------------------------------------- |
| **OS**               | Windows host; project lives in **WSL Ubuntu**                                           |
| **Project root**     | `/home/ubuntu/genai_pdn`                                                                      |
| **Cursor workspace** | `\\wsl.localhost\Ubuntu\home\ubuntu\gan`                                                |
| **Python**           | `.venv/bin/python` (always use venv, not system `python3`)                              |
| **GPU**              | Quadro GV100 32 GB; training often pinned to **physical GPU 1**                         |
| **Shell tip**        | Run commands via `wsl bash -c "..."` from PowerShell; avoid `<<` heredocs in PowerShell |


### Key paths

```
/home/ubuntu/genai_pdn/
├── .venv/                          # Python venv
├── data/heatmaps/all_combinations.csv   # 19,499 subsampled layouts (of ~2^52 full space)
├── data_multi_norm_robust/         # exp048 dataset (clipped z)
├── data_multi_norm_unbounded/      # exp049/050 dataset (unbounded z)
├── pipelines/normalize/multifreq.py
├── src_vae/others/norm_stats.py
├── experiments/exp048|exp049|exp050/
└── docs/
```

### Dataset scale (important context)

- Full simulation space: **~2^52 layout combos × 1–600 MHz** (user’s original iteration).
- Training data: **19,499 layouts × 16 MHz bins = 311,979 manifest rows**.
- Inverse-K subsampling: ~1000 layouts at K=2, tapering to ~150 at K=46–50.
- **400 MHz generalization is data-limited**, not just a loss/architecture issue.

---

## 2. Cursor / agent constraints

### `.cursorignore` (agents cannot read/edit these directly)

```
experiments/
datasets/
data_multi_norm*/
data/
evaluation/
.venv/
checkpoints/
**/*.pt, **/*.pth, **/*.ckpt, **/*.peb, **/*.h5
```

**Workaround:** use `wsl cat`, `wsl python3`, or patch scripts under `scrap/` then run via WSL.

### Docs index


| Doc                                                                                   | Purpose                                        |
| ------------------------------------------------------------------------------------- | ---------------------------------------------- |
| NORMALIZATION_AND_LOSS_STRATEGIES.md (`./NORMALIZATION_AND_LOSS_STRATEGIES.md`)        | Norm + loss theory, exp048→050 comparison      |
| TRAINING_GPU_UTILIZATION.md (`./TRAINING_GPU_UTILIZATION.md`)                          | CPU/GPU bottleneck fixes                       |
| experiments/exp050/EXPERIMENT_EXP050.md (`../experiments/exp050/EXPERIMENT_EXP050.md`) | exp050 runbook (in ignored dir; use `wsl cat`) |


---

## 3. Experiment arc (exp048 → exp049 → exp050)

Goal: improve **layout → heatmap** quality at high MHz (270/400), especially magnitude and spatial pattern.

### exp048 — clipped robust norm + heavy loss stack


|             |                                                                                                   |
| ----------- | ------------------------------------------------------------------------------------------------- |
| **Dataset** | `data_multi_norm_robust`                                                                          |
| **Norm**    | `robust_log1p_per_mhz` — per-MHz median/IQR, **z clipped** to p0.5–p99.5                          |
| **Losses**  | Full stack: huber, grad, lap, contrast, bg, dynrange, pearson, p99 phys, peak_loc, layout_sharpen |
| **Finding** | Clipped targets **flatten peaks** → poor spatial learning at high MHz                             |


### exp049 — unbounded norm, same loss stack


|             |                                                                          |
| ----------- | ------------------------------------------------------------------------ |
| **Dataset** | `data_multi_norm_unbounded`                                              |
| **Norm**    | `robust_log1p_per_mhz_unbounded` — same stats, **no FG z-clip**          |
| **Losses**  | Same heavy stack as exp048 + p99 phys                                    |
| **Finding** | Shape improved but **400 MHz Ω blow-up** (gen_max ~67 Ω, max_ratio 4.71) |


**Build unbounded dataset:**

```bash
cd /home/ubuntu/genai_pdn
NORM_ROBUST_PER_MHZ=1 NORM_UNBOUNDED_Z=1 .venv/bin/python pipelines/normalize/multifreq.py
```

### exp050 — Tier A lean losses (current best)


|               |                                                                                  |
| ------------- | -------------------------------------------------------------------------------- |
| **Dataset**   | `data_multi_norm_unbounded`                                                      |
| **Code**      | `experiments/exp050/codes/` (fork of exp038 training stack)                      |
| **Key files** | `heatmap_peak_losses.py`, `train_vae_simple.py`, `run_epoch_encode.py`           |
| **Training**  | Completed **700 epochs**; final ckpt: `checkpoint_epoch_700.pt`, `last_model.pt` |


#### Tier A active losses only


| Term           | Config key                           | Role                                       |
| -------------- | ------------------------------------ | ------------------------------------------ |
| FG huber       | (in `heatmap_loss_tier_a`)           | Full foreground anchor                     |
| Grad vector    | `heatmap_grad_vector_weight: 2.5`    | FG-masked Sobel Gx/Gy                      |
| Grad direction | `heatmap_grad_direction_weight: 1.5` | Flow angle where |∇T|>0.08                 |
| Phys Ω blob    | `heatmap_peak_phys_weight: 3.0`      | Top-24 target pixels, under+overshoot in Ω |
| peak_loc       | `heatmap_peak_loc_weight: 0.75`      | Hotspot position                           |
| latent_distill | `latent_distill_weight: 2.0`         | Layout teacher                             |


Scaled by `heatmap_weight: 5.5`. Focus phase: `heatmap_focus_heatmap_weight: 6.5`.

**Removed from exp050 code (not trained):** percentile/p99 losses, z dynrange blob, intensity peak, lap/contrast/bg, layout_sharpen, **Pearson in training** (kept for eval only).

#### Training history (exp050)

1. Initial run to epoch 500.
2. Resume from **epoch 300** with raised heatmap weights → extended to **epoch 700**.
3. User runs training manually on **GPU 1** via `run_train_gpu1.sh`.

#### Final config highlights (`experiments/exp050/config.yaml`)

```json
"num_epochs": 700,
"resume_checkpoint": "/home/ubuntu/genai_pdn/experiments/exp050/checkpoints/checkpoint_epoch_300.pt",
"data_dir": "/home/ubuntu/genai_pdn/data_multi_norm_unbounded",
"batch_size": 160,
"cache_in_ram": true,
"num_workers": 0,
"compile": false,
"layout_train_prob": 0.55
```

---

## 4. exp050 evaluation results (epoch 700, final)

Sweep: `experiments/exp050/multifreq_heatmap_sweep_30/`  
Checkpoint: `last_model.pt` | Mode: `layout_qc` | K=30 | n=2 val samples per anchor

### vs exp049 (same QC setup)


| MHz | Metric           | exp049     | exp050                  |
| --- | ---------------- | ---------- | ----------------------- |
| 270 | pearson_r        | 0.960      | **0.970**               |
| 270 | max_ratio        | 1.88       | **0.94**                |
| 400 | pearson_r        | 0.539      | **0.589** (still <0.75) |
| 400 | max_ratio        | **4.71**   | **1.56**                |
| 400 | gen_max (layout) | **66.7 Ω** | **9.2 Ω**               |


### Final exp050 QC summary


| MHz | pearson_r | max_ratio | Notes                                                       |
| --- | --------- | --------- | ----------------------------------------------------------- |
| 10  | 0.920     | 0.98      | Good shape; low-freq magnitude saturation near clip ceiling |
| 270 | 0.970     | 0.94      | **Strong** — essentially solved for this QC                 |
| 400 | 0.589     | 1.56      | Magnitude fixed; **spatial pattern still weak**             |


**Conclusion:** Tier A + unbounded norm fixed 400 MHz blow-up and improved 270 MHz. **400 MHz spatial generalization likely plateaued** on current ~19.5k layouts. Next lever is **targeted data** (more high-K layouts, denser 350–450 MHz), not more epochs on same data.

---

## 5. Training on GPU 1

### Run script

`experiments/exp050/run_train_gpu1.sh`:

```bash
export CUDA_VISIBLE_DEVICES=1   # physical GPU 1 → cuda:0 in PyTorch
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TORCH_CPU_THREADS=8
cd /home/ubuntu/genai_pdn
nohup .venv/bin/python -m experiments.exp050.codes.train_vae_simple > experiments/exp050/logs/train_gpu1_*.log 2>&1 &
```

See `experiments/exp050/GPU1_TRAINING.md` for full notes.

### Manual train / resume

```bash
cd /home/ubuntu/genai_pdn
export CUDA_VISIBLE_DEVICES=1
.venv/bin/python -m experiments.exp050.codes.train_vae_simple
```

Set `resume_checkpoint` in `config.yaml` to epoch int or full `.pt` path. Set `null` for fresh start.

### Monitor

```bash
tail -f experiments/exp050/metrics/loss.csv
tail -f experiments/exp050/logs/train_gpu1_*.log
watch -n2 nvidia-smi
```

---

## 6. Performance fixes (applied across training stack)

From `docs/TRAINING_GPU_UTILIZATION.md` and exp050 GPU setup:


| Fix              | Setting / change                                                               |
| ---------------- | ------------------------------------------------------------------------------ |
| RAM cache        | `cache_in_ram: true`, `num_workers: 0` — 311k samples preloaded (~80s startup) |
| No torch.compile | `compile: false` — GV100 does not benefit                                      |
| FP16             | `amp: fp16`, `tf32: false`                                                     |
| Batch size       | 160 (was 96 in older exps)                                                     |
| CPU threads      | `OMP/MKL/TORCH_CPU_THREADS=8` in shell + `torch.set_num_threads(8)` in train   |
| Loss sync        | Accumulate losses on GPU; single `.item()` per epoch (exp038 base)             |
| Quantile         | Replaced slow `torch.quantile` with `kthvalue` where still used                |
| empty_cache      | `empty_cache_interval: 0` in exp050 (was 25 in exp042)                         |


---

## 7. Evaluation commands

```bash
cd /home/ubuntu/genai_pdn

# Multifreq heatmap sweep + QC report
.venv/bin/python -m experiments.exp050.codes.eval_real_data_sweep

# Spatial metrics (pearson_fg etc.)
.venv/bin/python -m experiments.exp050.codes.eval_spatial_metrics

# Train vs val diagnostics
.venv/bin/python -m experiments.exp050.codes.eval_train_vs_val
```

Outputs: `experiments/exp050/multifreq_heatmap_sweep_30/` (`sweep_qc_report.md`, `sweep_qc_metrics.json`).

---

## 8. Normalization quick reference

**Forward (training targets):**

```
z = (log1p(Ω) - median_mhz) / IQR_mhz
```

**Denorm (inference/QC):**

```
Ω = expm1(z × IQR_mhz + median_mhz)
```

- **exp048:** z clipped; `heatmap_z_clip_min/max` enforced.
- **exp049/050:** unbounded z; `clip_min/max` in JSON are QC metadata only.

Central implementation: `src_vae/others/norm_stats.py`.

---

## 9. Open items for next agent

### Done / closed

- [x] exp050 trained to epoch 700
- [x] 400 MHz magnitude blow-up fixed vs exp049
- [x] 270 MHz QC strong (r≈0.97)

### Not done / suggested next steps

1. **400 MHz spatial quality** (r≈0.59) — user believes **more layout data** needed in high-freq representation region, not more loss tuning on 19.5k layouts.
2. **Data expansion options:**
  - `pipelines/dataset/subsample_inverse_k.py` — understand/adjust K sampling
  - `active_learning_pi/` — add layouts where layout-decode @400 MHz fails
  - Denser MHz anchors 350–450 if simulation budget allows
3. **exp051?** — if continuing model work: stronger layout→heatmap path, not heavier magnitude losses on same data.
4. **Config cleanup:** `exp050/config.yaml` may still contain legacy loss keys (lap, contrast, p99 weights) alongside Tier A keys; Tier A code in `heatmap_peak_losses.py` is source of truth for what actually runs.
5. **QC sample size:** sweep uses n=2 per anchor — noisy; consider more val layouts for reporting.
6. `**.cursorignore`:** editing `experiments/` requires WSL shell or `scrap/` patch scripts.

### User preferences (from rules)

- Do not commit unless asked.
- When implementing + running scripts, add/update `.md` documentation.
- Ask before deleting files.
- User trains manually — kill stray processes before they start a new run.

---

## 10. Quick command cheat sheet

```bash
# Enter project
cd /home/ubuntu/genai_pdn

# Train exp050 on GPU 1
export CUDA_VISIBLE_DEVICES=1
.venv/bin/python -m experiments.exp050.codes.train_vae_simple

# Or background
./experiments/exp050/run_train_gpu1.sh

# Kill stray training
pkill -f 'experiments.exp050.codes.train_vae_simple'

# Rebuild unbounded dataset
NORM_ROBUST_PER_MHZ=1 NORM_UNBOUNDED_Z=1 .venv/bin/python pipelines/normalize/multifreq.py

# Read ignored experiment files
wsl cat /home/ubuntu/genai_pdn/experiments/exp050/config.yaml
```
