---
title: training-procedure
type: concept
source: docs/training-procedure.md
tags: [concept, thesis]
---

> [!info] Mirror of `docs/training-procedure.md` — edit the source file, then re-run `tools/build_vault.py`.

# Training procedure (exp054 lineage → exp057)

Condensed description of the training loop used by the self-contained multi-input VAE trainers. Written from the exp054 process; **exp055–exp057** keep the same loop structure with architecture and loss deltas documented in [[model-architecture|model-architecture.md]].

Exact epoch counts, weights, and off-anchor MHz lists are config-specific — always cite the experiment’s `config.yaml` in the thesis.

---

## 1. Launch → entry

```text
run_train_*.sh
  └─ train_vae_simple.main()
       └─ train_vae()          # experiment patches
            └─ train_core.train_vae()
```

Typical: pin `CUDA_VISIBLE_DEVICES`, use `.venv/bin/python`, log under `logs/`.

---

## 2. One-time setup

1. Load `config.yaml`; map curriculum `*_frac` → absolute epochs when applicable.
2. Fresh run: clear old `checkpoints/`, `logs/`, `metrics/` (destructive — intentional for clean experiments).
3. Load `normalization_stats.json` / unbounded norm bundle; set FP32 / AMP policy.
4. Build multifreq dataloaders:
   - Layout store required
   - **Split by design** (e.g. 90/10)
   - Train: indexed subset with **cross-freq alt-MHz pairs**
   - Fixed draws per epoch (e.g. 30,000 with replacement) unless config says otherwise
5. Build model (`MultiInputVAEPoeFreq` family; exp057 adds structured latent + GNNs).
6. Optional resume from `checkpoint_epoch_*.pt` or `last_model.pt`.

### Optimizer pattern

- AdamW; often a **separate impedance LR** after an impedance-focus epoch (e.g. 80).
- `ReduceLROnPlateau` on validation loss when val runs.

---

## 3. Per-epoch structure

| Phase | Typical behavior |
|-------|------------------|
| Schedule scalars | β (KL) anneal, modality dropout, physics warmup, heatmap-focus phase weights |
| Train pass | Encode vs layout-path mix; full loss stack; grad clip; skip non-finite batches |
| Validation | Epoch 1, every \(N\) epochs (e.g. 25), and final — when `val_on_checkpoint_only` |
| Logging | `metrics/loss.csv`, impedance/heatmap split CSVs, timing |
| Checkpoint | Every \(N\) epochs → `checkpoint_epoch_{N}.pt` |
| Off-anchor eval | Every \(M\) epochs → `metrics/off_anchor_eval.csv` |

### Train forward (conceptual)

1. Prepare batch (H2D, FG mask, optional z guards).
2. Encode heatmap path → posterior \(z\), optional U-Net skips.
3. With probability `layout_train_prob`, train the **layout** decode path (deployment path) with optional latent/output distillation toward the encode teacher.
4. Apply PI frequency jitter on decode with configured probability.
5. Accumulate losses (heatmap, occupancy, impedance, KL, optional physics, cross-freq, cross-modal).

### Cross-frequency training

From a configured epoch onward, decode at an alternate MHz and compare to the paired alt heatmap — improves off-anchor behavior without requiring every MHz in every batch.

---

## 4. Illustrative schedule (400-epoch exp054 template)

| Epoch | Event |
|------:|-------|
| 20 | Cross-freq loss on |
| 80 | Impedance peak focus + separate impedance LR |
| ~240 | β anneal complete |
| ~260 | Heatmap-focus phase; protect heatmap in modality dropout |
| every 25 | Checkpoint + val |
| every 50 | Off-anchor eval append |

Later runs may extend epochs (e.g. 600) or AL fine-tunes add +50 epochs on overlay data — treat those as separate experimental stages in the thesis.

---

## 5. Output artifacts

| Path | Role |
|------|------|
| `checkpoints/checkpoint_epoch_*.pt` | Periodic snapshots |
| `checkpoints/last_model.pt` | Final / resume point (also AL fine-tune base) |
| `metrics/loss.csv` | Train/val core losses |
| `metrics/heatmap_peak_split.csv` | Peak/valley / Tier-A breakdown |
| `metrics/impedance_split.csv` | Impedance term breakdown |
| `metrics/off_anchor_eval.csv` | Off-anchor spatial metrics |
| `metrics/plots/*.png` | Convergence / component plots |

---

## 6. Thesis notes

- State **data split** (by design, not by row) to avoid leakage across MHz of the same layout.
- State **normalization mode** (unbounded robust log1p per-MHz) and **K≤30** scope.
- Do not mix checkpoint epochs from different architecture families (exp054≠055≠056≠057).
- For AL fine-tunes, report base epoch, overlay weight, and extra epochs ([[active-learning|active-learning.md]]).


## Implemented by

- [[experiments.exp059_capacity_freq.codes.train_core]] — `experiments/exp059_capacity_freq/codes/train_core.py`
- [[experiments.exp059_capacity_freq.codes.train_vae_simple]] — `experiments/exp059_capacity_freq/codes/train_vae_simple.py`
- [[experiments.exp059_capacity_freq.codes.distributed_train]] — `experiments/exp059_capacity_freq/codes/distributed_train.py`
- [[experiments.exp059_capacity_freq.codes.training_guard]] — `experiments/exp059_capacity_freq/codes/training_guard.py`
