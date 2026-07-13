# Fresh start v4 — layout-z / sweep alignment

## Root cause (from `scratch/diagnose_sweep_vs_dataset.py`)

| Path | mean r | mean p95 |
|------|--------|----------|
| sweep / layout @ π_ref=200 | ~0.22 | ~1 Ω |
| encode (sees GT heatmap) | ~0.92 | ~6.5 Ω |

The decoder works; **layout-z cross-freq decode does not**. Training spent 95% of steps on layout-z **without U-Net skips**, while the model only learned spatial detail on the encode path.

**Train/sweep mismatch:** cross-freq loss encoded layout at **native PI_freq**; sweep encodes at **π_ref=200 MHz** then decodes at sweep MHz.

## v4 changes

### Training code (`train_vae_simple.py`)

| Feature | Purpose |
|---------|---------|
| `cross_freq_layout_pi_ref_mhz: 200` | Cross-freq: `encode_layout(occ,imp @ 200MHz)` → decode @ `PI_freq_alt` — **matches sweep** |
| `layout_distill_weight: 2.0` | Every batch: layout-z decode should match posterior-z decode at same MHz |

### Config (`config.yaml`)

| Key | v3 | v4 | Why |
|-----|----|----|-----|
| `layout_train_prob_late` | 0.95 | **0.60** | More encode+skip supervision |
| `layout_train_phase_epoch` | 150 | **250** | Longer encode-heavy early phase |
| `cross_freq_weight` | 1.0 | **2.5** | Stronger off-anchor layout path |
| `layout_distill_weight` | — | **2.0** | Transfer spatial info to layout z |
| `cross_freq_layout_pi_ref_mhz` | — | **200** | Align with `PI_REF_MHZ` in sweep |
| `heatmap_phys_p99_weight` | 6.0 | **3.0** | Less conservative collapse |
| `heatmap_phys_p99_over_weight` | 2.0 | **0.75** | Allow peaks on layout path |
| `heatmap_spread_weight` | 3.5 | **5.0** | Penalize flat blue fields |
| `heatmap_hotspot_weight` | 4.0 | **6.0** | Localized peaks |
| `heatmap_private_dim` | 12 | **16** | More heatmap-specific capacity |
| `synthetic_blend_prob` | 0.12 | **0.18** | More between-anchor exposure |
| `resume_checkpoint` | 125 | **null** | Fresh start |

Keep: **log1p_gmax**, **v3 arch** (U-Net skips + occ spatial decoder).

## Start training

```bash
cd /home/ubuntu/gan
.venv/bin/python experiments/exp043/codes/train_vae_simple.py
```

Outputs → `experiments/exp043/runs/run_<UTC>/`

## Monitor during training

At checkpoints, watch off-anchor eval:

- `layout_cross @ 300 MHz` — target MSE ↓, phys p95 ↑
- `encode_cross` — should stay low (sanity)

After epoch ~200, run:

```bash
.venv/bin/python scratch/diagnose_sweep_vs_dataset.py \
  --checkpoint experiments/exp043/runs/run_<UTC>/checkpoints/last_model.pt
```

Target: `sweep_anchor_blend` mean r **> 0.5**, mean p95 **> 4 Ω**.

## If still flat after v4

1. Lower `layout_train_prob_late` to **0.45**
2. Raise `layout_distill_weight` to **3.5**
3. Add `cross_freq_pi_ref_only_prob` (always use π_ref for cross-freq) — not needed yet if `cross_freq_layout_pi_ref_mhz` is set
4. Architecture: 12×12 bottleneck or heatmap←layout cross-attention (heavier)
