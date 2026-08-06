---
title: EXP054_FIX_MISSING_CONFIG_FIELDS (archived)
type: archive
source: docs/_archive/EXP054_FIX_MISSING_CONFIG_FIELDS.md
tags: [archive]
---

> [!caution] Archived implementation note — not for thesis citation.

# exp054: Fix missing Config fields (AttributeError)

This note records the fix for repeated `AttributeError: 'Config' object has no attribute ...` during exp054 startup.

---

## 📝 Summary of Changes

- Added missing exp054 `Config` fields in `experiments/exp054_K_30/codes/train_vae_simple.py`:
  - Training guards: `training_recon_z_soft_floor`, `training_recon_z_soft_cap`, `training_recon_z_cap_mult`, `training_loss_finite_cap`
  - Optimizer toggle: `impedance_separate_lr`
  - Heatmap tier_a weights: `heatmap_pearson_weight`, `heatmap_grad_vector_weight`, `heatmap_grad_direction_weight`, `heatmap_grad_direction_min_mag`, `heatmap_grad_huber_delta`, `heatmap_peak_centroid_top_q`, `heatmap_valley_centroid_bottom_q`
- Mirrored those keys into `experiments/exp054_K_30/config.yaml` so yaml-override and `Config()` defaults stay aligned.

---

## 🚀 Implementation Details

### Why these errors happened

exp054 uses a vendored trainer (`train_core.py`) plus an exp054-specific entrypoint (`train_vae_simple.py`) that subclasses the base `Config`.

Some modules (notably `training_guard.py` and `heatmap_peak_losses.py`) access configuration values directly as `c.<name>`.
If a key is absent from the dataclass, Python raises `AttributeError` at runtime.

### What the new knobs do

- `training_recon_z_*` and `training_loss_finite_cap`: clamp/sanitize recon and loss tensors so NaN/Inf doesn’t explode training.
- `impedance_separate_lr`: allow a separate optimizer param group for impedance modules starting at `impedance_peak_focus_epoch`.
- Heatmap tier_a weights: required by Pearson/grad + centroid components of `heatmap_peak_losses.py`.

---

## 🛠️ Verification & Execution Results

```bash
# Verify Config construction
python3 -c "from experiments.exp054_K_30.codes.train_vae_simple import Config; c=Config(); print(c.impedance_separate_lr)"

# Smoke run: ensure no immediate AttributeError / import error
timeout 25 python experiments/exp054_K_30/codes/train_vae_simple.py
```

- Smoke run shows **no immediate** `AttributeError` / `ModuleNotFoundError`.
- Full training run not executed as part of this fix.
