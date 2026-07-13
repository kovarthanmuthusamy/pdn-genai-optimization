# exp043 heatmap v2 (post run_20260616T120139Z)

Training completed with v1 structural changes (8×8 bottleneck, pattern/spread losses). Visual eval still showed:

| Issue | Symptom |
|-------|---------|
| Magnitude overshoot | `gen_max` 2–10× `real_max` |
| Checkerboard | Rippling on vertical edges (ConvTranspose2d) |
| Blurred hotspots | Localized peaks smeared (lap + weak peak focus) |

Loss metrics looked good (`heatmap_loss` final ≈ 0.44) because clipping happened only in the **loss path**, not at **decode/inference**.

## v2 changes (fresh restart required)

### Decoder (`vae_multi_input_simple.py`)

1. **UpsampleConv2d** — bilinear upsample + 3×3 conv replaces all `ConvTranspose2d` in the heatmap decoder (anti-checkerboard).
2. **Sigmoid output bound** — `sigmoid(logits) * 1.02` at decode so inference and training share the same `[0, 1.02]` norm range.

### Losses (`gmax_heatmap_loss.py`)

| Loss | Purpose |
|------|---------|
| `heatmap_hotspot_weight` | Intensity²-weighted huber on high-Ω pixels |
| `heatmap_peak_centroid_weight` | Centroid match on top-8% target FG pixels |
| spread (updated) | Symmetric over-fill penalty on mean and p95 |

### Config (`config.yaml`)

| Key | v1 | v2 |
|-----|----|----|
| `heatmap_lap_weight` | 4.0 | **2.0** (less blur) |
| `heatmap_dynrange_over_weight` | 0.5 | **2.0** |
| `heatmap_phys_p99_over_weight` | 0.75 | **2.0** |
| `heatmap_phys_p99_weight` | 4.0 | **6.0** |
| `heatmap_hotspot_weight` | — | **4.0** |
| `heatmap_peak_centroid_weight` | — | **2.5** |

## Restart

```bash
cd /home/ubuntu/gan
.venv/bin/python scratch/_smoke_exp043_struct.py   # verify forward pass
.venv/bin/python experiments/exp043/codes/train_vae_simple.py
```

**Do not resume** `run_20260616T120139Z` checkpoints — decoder layer names and output activation changed.

## What to watch in compare plots

- `gen_max` should track `real_max` (within ~20%)
- No vertical edge rippling
- Localized corner/edge hotspots should appear in generated panel, not only smooth cyan fill
- Pattern diff `r` target: **> 0.85** on anchor MHz
