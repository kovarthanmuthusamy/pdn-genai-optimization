# exp047 Pipeline / QC Sweep Rules

## Purpose

The multifreq sweep is a **quality check** of the trained VAE before the next training
iteration — not a random layout explorer. Generation rules mirror
`pipelines/latent/optimize.py` and the production workflow.

## QC inference modes

| Mode | Layout source | Encode @ PI_REF | Decode @ sweep MHz | Use case |
|------|---------------|-----------------|--------------------|----------|
| **`layout_qc`** (default) | Val set at fixed K | `encode_layout(occ, imp)` | yes | Model QC after training |
| **`latent_z`** | `best_latent.npy` | no — z fixed | yes | Post latent-optimization |
| **`layout_hybrid`** | latent run + imp | `encode_layout_full` + shared←z_opt | yes | Post-opt with private head |

**Blocked when `QC_SWEEP=True`:** `marginal`, `anchor_blend`, `layout` without val layouts
(set `ALLOW_RANDOM_LAYOUT=True` to override).

## Hard rules (`scrap/generation/sweep_latent_opt_rules.py`)

1. Fixed layout — occ/imp from val or latent-run export, never marginal draw (unless override).
2. Layout encode uses `PI_REF_MHZ` (default 200 MHz); only the **decoder** sees each sweep MHz.
3. `latent_z` — load `best_latent.npy`, change `PI_freq` per peak only (matches optimize heatmap path).
4. `layout_hybrid` — private dims from layout head, shared dims from optimized z.
5. PEB occupancy — hard top-K (`topk_occ_binary`), same as optimize `_topk_occ`.

## Config (pipeline + generator)

```python
EXPERIMENT_DIR = "experiments/exp047"
DATA_DIR = "data_multi_norm"
QC_SWEEP = True
INFERENCE_MODE = "layout_qc"
LAYOUT_SOURCE = "val"
PI_REF_MHZ = 200.0
K_VALUE = 30
```

### Post latent-optimization QC

```python
INFERENCE_MODE = "latent_z"  # or "layout_hybrid"
LAYOUT_SOURCE = "latent_run"
LATENT_RUN_DIR = "data/latent_runs/exp047/0"  # contains K30/best_latent.npy
```

## Files changed

- `scrap/generation/sweep_latent_opt_rules.py` — shared rules + loaders
- `scrap/generation/run_multifreq_heatmap_sweep.py` — QC modes in `run_generate`
- `scrap/orchestration/run_multifreq_sweep_pipeline.py` — CONFIG passthrough
- `experiments/exp047/codes/inference_vae.py` — `use_layout_private_head` on load

## Outputs after generate

| File | Purpose |
|------|---------|
| `sweep_qc_report.md` | Human-readable QC + embedded copy block |
| `sweep_qc_metrics.json` | Structured metrics for tooling |
| Terminal | Prints `=== SWEEP QC (copy to agent) ===` block |

### Metrics included

1. **Per sweep MHz** — gen_max, mean_fg, p95, near-clip flag (magnitude saturation)
2. **Training anchors** (`layout_qc` only) — pearson_r, fg_mse, mae_phys, real_max vs gen_max
3. **Auto flags** — WARN if pearson &lt; 0.75, max_ratio &gt; 1.5, or gen_max near z-clip ceiling

Set `RUN_SWEEP_QC_EVAL = False` to skip.

## Still separate

- `pipelines/latent/optimize.py` — still defaults to exp038; update when moving latent opt to exp047.
- `eval_real_data_sweep.py` — metric QC on val layouts (Pearson r, FG-MSE); run alongside sweep for numbers.
