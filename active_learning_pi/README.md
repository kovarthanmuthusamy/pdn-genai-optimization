# Active learning PI heatmaps

Pipeline from [docs/active-learning-pi-heatmaps.md](../docs/active-learning-pi-heatmaps.md):

**Infer everything first (cheap) → simulate only the worst cases once (expensive) → fine-tune.**

## Workflow

```
generate candidates (48×)
    ↓
infer + score each candidate one-by-one (MC dropout, no ECADStar)
    ↓
select worst only (high uncertainty / low quality_score)
    ↓
one combined .peb  →  one Load Batch in PI/EMI
    ↓
ingest real heatmaps
    ↓
normalize (training stats from data_multifreq_norm)
    ↓
fine-tune model
```

Good candidates (low uncertainty) are **not** sent to ECADStar.

## Commands

From `gan/`:

```bash
# Full cycle (recommended)
python active_learning_pi/run_pipeline.py cycle

# Or step by step
python active_learning_pi/run_pipeline.py generate   # new iter, candidate pool
python active_learning_pi/run_pipeline.py infer      # score all, one-by-one
python active_learning_pi/run_pipeline.py select-bad
python active_learning_pi/run_pipeline.py build-peb  # single batch_simulate_once.peb
python active_learning_pi/run_pipeline.py simulate  # Windows + AutoHotkey
python active_learning_pi/run_pipeline.py ingest
python active_learning_pi/run_pipeline.py normalize
python active_learning_pi/run_pipeline.py evaluate
```

Windows:

```powershell
.\active_learning_pi\scripts\run_iteration.ps1
```

(Edit script to call `cycle` instead of `iteration`.)

## Config (`config/default.json`)

| Key | Meaning |
|-----|---------|
| `num_candidates` | How many layouts×MHz to **infer** (cheap) |
| `simulate_batch_size` | How many **worst** go into the one `.peb` |
| `min_uncertainty_percentile` | Only simulate if badness ≥ this percentile (default 50) |
| `mc_passes` | Stochastic passes per candidate for uncertainty |
| `normalization_stats_json` | Training stats (default: `datasets/data_multifreq_norm/normalization_stats.json`) |

## Normalization (after simulation)

Uses the **same rules** as `scripts/Normalization.py`:

- **Heatmap**: log(1+x) z-score on foreground, background sentinel → `(1, 64, 64)`
- **Impedance**: log z-score → `(1, 231)` (from simulation CSV if present, else denormalized from model prediction)
- **Occupancy**: copied as float32 `(52,)`
- **PI_freq**: Hz float64 (unchanged)

Outputs per iteration:

- `raw_dataset/` — multifreq layout before normalize
- `dataset_norm/` — ready to merge into training / fine-tune

## Outputs

`active_learning_pi/runs/al_run_001/iter_0001/`:

- `candidates.json` — full pool
- `scored_candidates.json` — all inference scores
- `selected_for_simulation.json` — worst only
- `batch_simulate_once.peb` — single ECADStar batch
- `labels/sample_*/` — ground truth after ingest
- `dataset_norm/` — normalized training-ready tensors

## ECADSTAR automation

Uses `scripts/run_ecadstar_piemi_batch.ps1` + `scripts/ecadstar_piemi_batch.ahk`.  
Edit `ecadstar` paths in config.
