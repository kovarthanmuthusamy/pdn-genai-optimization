# Fresh start v5 — layout path alignment + teacher skips

## Problem (v4 outcome @ ep 600)

| Metric | encode_cross @ 300 MHz | layout_cross @ 300 MHz |
|--------|------------------------|------------------------|
| FG MSE | ~0.0004 | ~0.010 |
| Sweep compare | — | r ≈ 0.35, gen p99 ≈ 13 Ω vs real ≈ 26 Ω |

Encode path works; **layout-z cross-freq decode** still flat. v4 aligned π_ref for cross-freq loss but:
- Layout forward still encoded at **native MHz** (not π_ref)
- Layout decode had **no U-Net skips** (heatmap spatial detail discarded)
- Distill matched same-π only, not cross-freq sweep path

## v5 changes

### Code (`train_vae_simple.py`)

| Feature | Purpose |
|---------|---------|
| `layout_teacher_skips: true` | Pass GT heatmap encoder skips into layout/cross-freq decode (**train only**) |
| `layout_encode_pi_ref: true` | Layout encode at **200 MHz** (π_ref), decode at native MHz — matches sweep |
| `layout_distill_cross_freq: true` | Distill: `layout@π_ref → decode@alt` toward `posterior → decode@alt` |
| Cross-freq loss + teacher skips | Stronger gradient on layout decode with spatial hints |

### Config (`config.yaml`)

| Key | v4 | v5 | Why |
|-----|----|----|-----|
| `heatmap_private_dim` | 16 | **20** (fixed) | Heatmap + freq PoE private tail |
| `latent_dim` | 42 | **55** (shared **35** + private **20**) | More shared capacity for occ/imp/layout z |
| `cross_freq_weight` | 2.5 | **4.0** | Pressure layout cross-freq path |
| `heatmap_focus_cross_freq_weight` | 2.0 | **3.5** | Same in heatmap-focus phase |
| `layout_train_prob_late` | 0.60 | **0.40** | More encode+skip batches (60% encode path) |
| `layout_distill_weight` | 2.0 | **2.5** | Stronger transfer to layout z |
| `layout_teacher_skips` | — | **true** | Train-time skip injection |
| `layout_encode_pi_ref` | — | **true** | Sweep-aligned layout encode |
| `layout_distill_cross_freq` | — | **true** | Cross-freq distill |
| `resume_checkpoint` | 475/600 | **null** | **Fresh start** (private dim 16→20 breaks weight shapes) |

### Eval (`eval_cross_freq_gmax.py`)

`layout_cross` now encodes layout at **π_ref=200 MHz** (was native MHz) — matches sweep metric.

## Will `heatmap_private_dim = 20` help?

**Modestly, not magically.**

- Private dims are filled by the **heatmap expert** (encode path) and the **freq PoE expert** (private tail on layout z).
- Layout path still **never runs the heatmap encoder** — so extra private dims do not add GT spatial information directly.
- Benefits:
  - Slightly richer **freq-specific** latent coding on layout path (freq expert → 20-dim private tail).
  - Better **teacher** in distill (posterior has more heatmap-specific capacity).
- **Main v5 gains** come from teacher skips + π_ref encode + cross-freq distill, not private dim alone.

## Start training

```bash
cd /home/ubuntu/gan
.venv/bin/python experiments/exp043/codes/train_vae_simple.py
```

Outputs → `experiments/exp043/runs/run_<UTC>/`

## Monitor

At checkpoints, `metrics/off_anchor_eval_epoch_*.csv`:

- `layout_cross @ 300 MHz` — target MSE **< 0.003** (was ~0.010)
- `encode_cross` — should stay **< 0.001**

After epoch ~200:

```bash
.venv/bin/python scratch/diagnose_sweep_vs_dataset.py \
  --checkpoint experiments/exp043/runs/run_<UTC>/checkpoints/checkpoint_epoch_200.pt
```

Target: `sweep_anchor_blend` mean r **> 0.5**, phys p95 **> 4 Ω**.

## Note on teacher skips at inference

Teacher skips use **GT heatmap** — only available during training. At sweep inference, decode still runs **without skips**. Training with skips teaches the decoder sharper spatial reconstruction; layout z must still learn to carry enough signal when skips are absent.
