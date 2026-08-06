---
title: evaluation-metrics
type: concept
source: docs/evaluation-metrics.md
tags: [concept, thesis]
---

> [!info] Mirror of `docs/evaluation-metrics.md` — edit the source file, then re-run `tools/build_vault.py`.

# Evaluation metrics

Primary metrics for judging generated PI heatmaps against ECADStar (or held-out) references, and for off-anchor surrogate evaluation.

---

## 1. Design principles

1. **Do not rely on raw `max()` or global `mae_ohm` alone** — low-MHz maps often have near-zero MAE while high-MHz peaks dominate design risk.
2. Prefer **foreground** statistics (masked IC/PDN region).
3. Separate **shape** agreement from **peak scale** agreement.
4. Report **off-anchor** frequencies (between training anchors) for the **layout** decode path used at deployment (`layout_cross`).

---

## 2. Sim-compare primary QC

Used in multifreq sweep compare reports (`scrap/comparison/heatmap_sim_metrics.py`, orchestration docs).

| Rank | Metric | Interprets |
|------|--------|------------|
| 1 | `pearson_r` | Spatial pattern correlation (shape) |
| 2 | `pattern_mae` | Pattern error after scale-insensitive alignment |
| 3 | `max_ratio` | `gen_max / real_max` (peak scale) |

Secondary (still logged, deprioritized in summaries): `mae_ohm`, `mape_pct` — unstable or uninformative at some MHz (e.g. `mae_ohm≈0` at 10 MHz).

**Pros:** Matches engineering judgment (shape + peak height).  
**Cons:** Pearson is insensitive to uniform scale; always pair with `max_ratio`.

---

## 3. Off-anchor surrogate metrics

During training and AL cycles:

| Metric family | Typical use |
|---------------|-------------|
| Foreground p95 / p99 MAE (Ω or log1p) | Peak-region error |
| `layout_cross` error at off-anchor MHz | Deployment path |
| `encode_cross` (if logged) | Teacher/encode path comparison |
| Peak location error | Hotspot position |

AL cycle reports emphasize **pre vs post fine-tune** ECAD p99 MAE on ingested layouts, plus training off-anchor tables at the final fine-tune epoch.

Default AL eval MHz examples: **90, 265, 435, 510** (between anchors).

---

## 4. What not to claim from a single metric

| Observation | Risk |
|-------------|------|
| High Pearson, bad `max_ratio` | Shape OK, magnitude wrong |
| Low MAE at low MHz only | Understates high-MHz failure |
| Train loss ↓ without ECAD re-sim | Overlay overfit / distribution shift |
| AL improvement without random baseline | Selection bias / confounding |

---

## 5. Related artifacts

- Sweep metrics CSV / MD under experiment sweep folders
- `metrics/off_anchor_eval.csv` during training
- AL: `eval_off_anchor_*.json`, `CYCLE_EVAL_REPORT.md` (see [[active-learning|active-learning.md]])


## Implemented by

- [[experiments.exp059_capacity_freq.codes.spatial_metrics]] — `experiments/exp059_capacity_freq/codes/spatial_metrics.py`
- [[experiments.exp059_capacity_freq.codes.eval_spatial_metrics]] — `experiments/exp059_capacity_freq/codes/eval_spatial_metrics.py`
- [[experiments.exp059_capacity_freq.codes.eval_off_anchor]] — `experiments/exp059_capacity_freq/codes/eval_off_anchor.py`
- [[evaluate_cycle]] — `active_learning_pi/al/evaluate_cycle.py`
