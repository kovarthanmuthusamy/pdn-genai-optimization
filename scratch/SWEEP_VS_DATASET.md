# Sweep vs Dataset Diagnostic (exp043 v3 + log1p)

Script: `scratch/diagnose_sweep_vs_dataset.py`

## Purpose

Separates **three inference paths** on real validation layouts at K=30, 300 MHz:

| Path | What it does |
|------|----------------|
| `sweep_anchor_blend` | Same as multifreq sweep: marginal z → gen occ/imp → layout-z @ π_ref=200 → anchor_blend decode @ 300 MHz |
| `layout_gt` | True dataset occ/imp, but still encode layout @ π_ref=200 (sweep layout path) |
| `layout_gt_native_pi` | True occ/imp, encode layout @ **native 300 MHz** (matches `eval_cross_freq_gmax`) |
| `encode_gt` | Full encode with GT heatmap + occ + imp @ 300 MHz (upper bound) |

## Run

```bash
.venv/bin/python scratch/diagnose_sweep_vs_dataset.py
.venv/bin/python scratch/diagnose_sweep_vs_dataset.py --k 30 --mhz 300 --num-samples 5 --no-plots
```

Checkpoint default: `experiments/exp043/runs/run_20260616T165326Z/checkpoints/last_model.pt`

Outputs:
- `scratch/sweep_vs_dataset_report.json`
- `scratch/sweep_vs_dataset_summary.txt`
- `scratch/sweep_vs_dataset_plots/` (unless `--no-plots`)

## Results (epoch 600, K=30, 300 MHz, 5 val samples)

| Path | mean Pearson r | mean phys p95 |
|------|----------------|---------------|
| `sweep_anchor_blend` | **0.22** | **0.99 Ω** |
| `layout_gt` | 0.29 | 0.97 Ω |
| `layout_gt_native_pi` | 0.15 | 2.05 Ω |
| `encode_gt` | **0.92** | **6.45 Ω** |

## Conclusions

1. **Blue heatmaps in compare are real** — sweep path predicts ~1 Ω at p95 vs GT ~6–15 Ω; matches `gen_p95≈1` in compare plots.

2. **Training loss looking good is misleading** for this visual — `encode_gt` MSE ≈ 0.001 (excellent) but **layout-z decode fails** (MSE ≈ 0.07, r ≈ 0.2). Training mostly uses layout-z (95%) without U-Net skips.

3. **Generated occ/imp is NOT the main gap** — `sweep_anchor_blend` ≈ `layout_gt` with **true** occ/imp. Marginal layout generation adds only ~occ overlap 0.33 vs 0.54, not the full collapse.

4. **Bottleneck = layout-z cross-freq decode**, not log1p denorm or dataset skew alone. The heatmap decoder works when the heatmap encoder contributes (`encode_gt`).

5. **Saved sweep `.npy` vs random dataset GT** — r≈0.5 on sample 0 is not apples-to-apples; compare uses ECADStar sim on the **generated** layout, not dataset GT heatmaps.

## Suggested next steps

- Tune training: more encode/skip supervision, lower `layout_train_prob_late`, raise cross-freq weight with heatmap skips.
- Sweep validation: add `INFERENCE_MODE=encode` branch when GT layouts exist; or report layout-z vs encode gap explicitly.
- Soften `heatmap_phys_p99_over_weight` — may reinforce low-quantile collapse on layout path.
