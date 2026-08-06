---
title: EXP054_EXTEND_600_EPOCHS (archived)
type: archive
source: docs/_archive/EXP054_EXTEND_600_EPOCHS.md
tags: [archive]
---

> [!caution] Archived implementation note — not for thesis citation.

# exp054: Extend training 400 → 600 epochs

---

## 📝 Summary of Changes

- `num_epochs`: 400 → **600**
- `resume_checkpoint`: **"latest"**
- `recalculate_curriculum_on_resume`: **false** (keep milestone epochs from checkpoint)
- `reset_lr_on_resume`: **false** (continue current LR, do not jump back to base LR)
- Scaled all `*_frac` keys for a 600-epoch timeline (explicit `*_epoch` keys unchanged)
- Late focus weights bumped: `heatmap_focus_heatmap_weight` 4.0, `impedance` 1.75, `cross_freq` 1.1
- `lr_patience`: 10 → 15
- Added `run_train_extend_600.sh`

---

## 🚀 How to resume

1. **Stop** the current run (Ctrl+C) after the latest checkpoint, or let it finish epoch 400.
2. Confirm latest checkpoint exists:
   ```bash
   ls -t experiments/exp054_K_30/checkpoints/checkpoint_epoch_*.pt | head -1
   ```
3. Launch extension:
   ```bash
   ./experiments/exp054_K_30/run_train_extend_600.sh
   ```
   Or single GPU:
   ```bash
   NPROC=1 python experiments/exp054_K_30/codes/train_vae_simple.py
   ```

Training continues from `latest` → epochs **401–600** (or from last completed epoch + 1).

---

## Schedule (unchanged absolute epochs on resume)

| Milestone | Epoch |
|-----------|------:|
| Cross-freq | 20 |
| Impedance peak / LR split | 80 |
| β anneal done | 240 |
| Heatmap focus | 260 |
| **New end** | **600** |

All phases are already active at ~epoch 372; extension adds **200 epochs** in heatmap-focus phase with slightly higher hm/imp/cf weights.

---

## 🛠️ Verification

At startup you should see:
```
Resume: .../checkpoint_epoch_XXX.pt
  Curriculum restored from checkpoint (not re-scaled to new num_epochs)
  Reset LR on resume — NOT run (reset_lr_on_resume=false)
epochs XXX+1–600
```
