# EXP055 (hard_occ): Top-region Huber replaces centroid loss

### 📝 Summary of Changes

- **Fixed broken module paths** in `experiments/exp055_hard_occ/` — internal imports pointed at
  `experiments.exp055_K_30` (folder was renamed) and could not import. Now `experiments.exp055_hard_occ`.
- **Turned OFF centroid loss** for peak and valley heatmap extrema.
- **Added a top-region Huber term**: Huber on log1p amplitude, restricted to GT pixels above p95
  (peak) / below p05 (valley). This cares about both **peak position** (only extreme-region pixels
  count) and **amplitude** (Huber on the value), unlike the position-only centroid.

### 🚀 Implementation Details

#### New loss term (`codes/heatmap_peak_losses.py`)
```python
def _topregion_huber(recon_lp, tgt_lp, fg, *, side, c):
    delta = c.heatmap_topregion_huber_delta
    if side == "peak":
        # background filled with per-sample MIN so it never enters the TOP percentile
        thr = p95(tgt_lp over FG)
        mask = (tgt_lp >= thr) * fg
    else:
        # background filled with per-sample MAX so it never enters the BOTTOM percentile
        thr = p05(tgt_lp over FG)
        mask = (tgt_lp <= thr) * fg
    hub = huber(recon_lp, tgt_lp, delta)
    return (hub * mask).sum() / mask.sum()   # per-sample mean over top-region
```
- Registered as term `topregion` in `_TERMS` for both peak and valley.
- Operates in **log1p physical space** (same as centroid / extrema terms).
- Bounded cap 4.0 (like centroid) via `_bundle`.

#### Why it beats centroid here
| | Centroid (old) | Top-region Huber (new) |
|--|----------------|------------------------|
| Position | Yes (weighted centroid of top region) | Yes (only top-region pixels contribute) |
| Amplitude | **No** (position only) | **Yes** (Huber on log1p value) |
| Peak location | Indirect | Direct (hot pixels must match) |

#### Config (`config.yaml`)
```json
"heatmap_peak_centroid_weight": 0.0,
"heatmap_valley_centroid_weight": 0.0,
"heatmap_peak_topregion_weight": 2.5,
"heatmap_valley_topregion_weight": 2.5,
"heatmap_peak_topregion_q": 0.95,
"heatmap_valley_topregion_q": 0.05,
"heatmap_topregion_huber_delta": 0.5,
"heatmap_peak_log1p_weight": 1.0,
"heatmap_valley_log1p_weight": 1.0
```
New `Config` dataclass fields added in `codes/train_vae_simple.py` with matching defaults
(topregion weights default 0.0 → opt-in via yaml).

The term flows through the existing `heatmap_extrema_log1p_both` path, so it is reported under
`heatmap_peak_log1p_loss` / `heatmap_valley_log1p_loss` in `metrics/heatmap_peak_split.csv`.

### 🛠️ Verification & Execution Results

**Unit test** (synthetic peak):
```
peak topregion: [0.0455, 0.0427]   valley topregion: [0.0420, 0.0454]
perfect recon peak total = 0.0      gradients finite ✓
```

**Training startup** (`timeout 60`):
```
python -m experiments.exp055_hard_occ.codes.train_vae_simple
→ EXP055_HARD_OCC — MULTI-INPUT VAE, 12.8M params, occ_bin=True
→ config loaded (182 keys), fresh 400-epoch run, no import/attribute errors
```

### Run
```bash
./experiments/exp055_hard_occ/run_train_gpu1.sh
# or DDP
./experiments/exp055_hard_occ/run_train_ddp.sh
```

Judge on `metrics/off_anchor_eval.csv` (peak_loc + mse) — the top-region Huber should tighten
peak amplitude at 270/400 MHz where centroid (position-only) could not.
