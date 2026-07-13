# exp053 — resume ep325 with boosted peak losses

**Mode:** resume from `checkpoint_epoch_325.pt`, run ep326→450 with stronger log1p peak stack.

```bash
cd /home/ubuntu/genai_pdn
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m experiments.exp053_peak_log1p_losses.codes.train_vae_simple
```

**Peak tweaks (vs ep1–325):**
- `heatmap_peak_log1p_weight`: 2.75 (was 1.5)
- `heatmap_peak_hotspot_weight`: 2.5 (was 2.0)
- `heatmap_peak_max_log1p_weight`: 1.25 (was 1.0)
- `heatmap_peak_centroid_weight`: 1.875 (was 1.5)
- `heatmap_mhz_loss_weight_max`: 3.0 (was 2.5)

**Metrics:** `metrics/heatmap_peak_split.csv` logs tier_a / peak_log1p / phys_blob per checkpoint.

See `docs/EXP053_FRESH_TRAINING.md`, `docs/EXP053_TRAINING_DIAG_EP350.md`.
