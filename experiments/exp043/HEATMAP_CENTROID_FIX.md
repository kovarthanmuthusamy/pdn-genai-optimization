# Heatmap centroid loss scale fix (exp043)

## Symptom

After log1p train-space + v3 architecture restart, epoch-1 logs showed:

```
loss=109749  hm=22893.96×4.0
```

Prior linear-norm runs reported `heatmap_loss ≈ 0.44` late in training.

## Root cause

`_fg_peak_centroid_loss` in `gmax_heatmap_loss.py` computed hotspot centroids in **raw pixel coordinates** (0–63). Squared L2 error per axis can reach ~4000, so with `heatmap_peak_centroid_weight=2.5` a single batch contributed **~20k+** to `heatmap_loss` at random init.

Other aux terms (pattern, spread, hotspot, grad/lap) were O(1–10).

## Fix

Normalize centroid coordinates to **[0, 1]** by dividing by `(H-1)` and `(W-1)` before the squared distance.

| Term | Before (random init) | After |
|------|---------------------|-------|
| centroid (unweighted) | ~1190 | ~0.26 |
| full heatmap loss (config weights) | ~22k (real training) | ~7.7 (diagnostic) |

## Verification

```bash
.venv/bin/python scratch/_breakdown_hm_loss.py
```

## Action

**Restart training** from scratch (`resume_checkpoint: null`). Do not resume checkpoints trained with the buggy centroid scale — loss landscape was wrong.

```bash
cd /home/ubuntu/gan
.venv/bin/python experiments/exp043/codes/train_vae_simple.py
```

Expected epoch-1 `hm` is single-digit to low tens, not thousands.
