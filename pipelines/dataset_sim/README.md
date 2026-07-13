# Combinations dataset simulation

End-to-end workflow to simulate **10,000 new decap layouts** in ECADStar and collect raw `PI-*` folders for downstream ingest (normalization, manifest build, training — handled by other scripts).

```
combinations.csv  →  .peb files  →  ECADStar  →  Impedance/ + heatmaps_{MHz}MHz/
```

---

## Overview

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1. Sample layouts | `pipelines/heatmaps/sample_new_combinations.py` | `all_combinations.csv` (19,499 existing) | `combinations.csv` (10,000 new) |
| 2. Simulate | `pipelines/dataset_sim/run_combinations_sim_pipeline.py` | `combinations.csv` | `datasets/data_combinations_sim/` |

Each CSV row is **52 values** (0/1) = one decap placement (C1–C52).  
Row index `i` corresponds to **`PI-(i+1)`** in simulation output when `START_LAYOUT=0`.

---

## Why one PEB per phase (not batches)

ECADStar creates output folders **in PEB script order**:

- 1st job in `.peb` → `PI-1`
- 2nd job → `PI-2`
- …
- 10,000th job → `PI-10000`

So this pipeline builds **one `.peb` file containing all layouts** for each phase. Splitting into smaller batches would reset numbering to `PI-1` each time and break alignment with `combinations.csv` row order.

---

## Step 1 — Create `combinations.csv`

```bash
cd /home/ubuntu/genai_pdn
python pipelines/heatmaps/sample_new_combinations.py
```

**What it does**

- Reads `data/heatmaps/all_combinations.csv` (layouts already simulated)
- Writes `data/heatmaps/combinations.csv` with **10,000 new** rows (no duplicates)
- **K=2:** all remaining unique pairs (~326 layouts)
- **K=3–50:** inverse-K sampling for the rest (more layouts at low K)

**Key parameters** (`sample_new_combinations.py` CONFIG):

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `TOTAL_N` | `10000` | Number of new layouts |
| `EXISTING_CSV` | `all_combinations.csv` | Layouts to exclude |
| `OUTPUT_CSV` | `combinations.csv` | Output file |
| `SEED` | `42` | Random seed |

Report: `data/heatmaps/combinations_sample_report.json`

---

## Step 2 — Run ECADStar simulation

```bash
python pipelines/dataset_sim/run_combinations_sim_pipeline.py
```

Edit the **CONFIG** block at the top of that file before running (paths, MHz list, skip flags).

### Phases (in order)

For **all** layouts in the selected CSV slice (default 10,000):

1. **Impedance** (`CreatePISpectrum`)
   - Build `peb/combinations_impedance.peb`
   - ECADStar Load Batch → wait → move `PI-1`…`PI-N` → `Impedance/`

2. **PI-Distribution** (one phase per MHz)
   - Build `peb/combinations_dist_{MHz}MHz.peb` (e.g. `combinations_dist_10MHz.peb`)
   - Simulate → move `PI-1`…`PI-N` → `heatmaps_{MHz}MHz/`

MHz list comes from `configs/multifreq_anchors.yaml` when `USE_ANCHOR_MHZ=True`, or from `SWEEP_MHZ` when `False`.

### Output layout

```
datasets/data_combinations_sim/
  peb/
    combinations_impedance.peb
    combinations_dist_10MHz.peb
    combinations_dist_63MHz.peb
    ...
    combinations_dist_400MHz.peb

  Impedance/
    PI-1/
    PI-2/
    ...
    PI-10000/

  heatmaps_10MHz/
    PI-1/ ... PI-10000/
  heatmaps_63MHz/
    PI-1/ ... PI-10000/
  ...
  heatmaps_400MHz/
    PI-1/ ... PI-10000/

  sim_progress.json
```

Each `PI-{n}` folder is the **raw ECADStar output** (maps, CSVs, etc.) moved unchanged from the `.emc` directory. No `layout_*` renaming — other scripts handle ingest.

---

## How the pipeline knows simulation is finished

After starting Load Batch, the script polls `ECADSTAR_EMC_OUTPUT_DIR` every `ECADSTAR_WAIT_POLL_SEC` seconds until:

| Phase | Ready when each `PI-{n}` has… |
|-------|-------------------------------|
| Impedance | At least one **`.csv`** with mtime after batch start |
| Heatmaps | A **`Z_*MHz.map`** file with mtime after batch start |

All `PI-1` through `PI-N` must be ready, or `ECADSTAR_WAIT_TIMEOUT_SEC` is hit (default **48 hours** per phase).

Then it moves folders to `Impedance/` or `heatmaps_{MHz}MHz/`.

---

## Configuration (`run_combinations_sim_pipeline.py`)

### Paths & data

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `COMBINATIONS_CSV` | `data/heatmaps/combinations.csv` | Input layouts |
| `OUTPUT_ROOT` | `datasets/data_combinations_sim` | Simulation output root |
| `PEB_DIR` | `{OUTPUT_ROOT}/peb` | Generated `.peb` files |
| `START_LAYOUT` | `0` | First CSV row (0-based) |
| `MAX_LAYOUTS` | `None` | Max rows; `None` = all after `START_LAYOUT` |

### Frequencies

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `USE_ANCHOR_MHZ` | `True` | Use `configs/multifreq_anchors.yaml` |
| `SWEEP_MHZ` | list | Used only if `USE_ANCHOR_MHZ=False` |
| `SKIP_MHZ` | `set()` | MHz values to skip, e.g. `{500}` |

### PEB / design

| Parameter | Default |
|-----------|---------|
| `POWERBUS` | `"Power_GND"` |
| `IMPEDANCE_COMPONENTS` | `"IC1_Port1,IC2_Port2"` |

### ECADStar (Windows paths — edit for your machine)

| Parameter | Default |
|-----------|---------|
| `PEB_COPY_DEST` | `C:\Users\...\Desktop\design\PEB` |
| `ECADSTAR_ERF_PATH` | `...\H-shape.erf` |
| `ECADSTAR_EMC_OUTPUT_DIR` | `...\H-shape.emc` |
| `ECADSTAR_AHK_EXE` | `None` (AutoHotkey default) |
| `ECADSTAR_SKIP_OPEN_ERF` | `False` | `False` (default) re-opens `.erf` every phase; `True` = Load Batch only |
| `ECADSTAR_CLEAR_LOCK_FILE` | `True` |
| `ECADSTAR_WAIT_TIMEOUT_SEC` | `172800` (48 h) |
| `ECADSTAR_WAIT_POLL_SEC` | `120` |
| `ECADSTAR_BATCH_START_TIMEOUT_SEC` | `900` | Max wait for PI-1 log to confirm batch started |
| `ECADSTAR_BATCH_START_POLL_SEC` | `10` | Poll interval during batch-start confirm |

### Step control

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `SKIP_IMPEDANCE` | `False` | Skip impedance phase |
| `SKIP_DISTRIBUTION` | `False` | Skip all heatmap MHz phases |
| `SKIP_SIMULATE` | `False` | `True` = only write `.peb`, no ECADStar |
| `SKIP_MOVE` | `False` | Do not move `PI-*` folders |
| `CLEAN_DEST_BEFORE_PASTE` | `True` | Clear existing `PI-*` in destination before move |
| `RESUME_FROM_PROGRESS` | `True` | Skip phases already in `sim_progress.json` |

### Resume file (`sim_progress.json`)

```json
{
  "impedance_done": true,
  "distribution_mhz_done": [10, 63, 80, 130, ...]
}
```

If a MHz step fails, fix ECADStar and re-run — completed steps are skipped.

---

## Example runs

### Test: build PEBs only (no ECADStar)

In CONFIG:

```python
SKIP_SIMULATE = True
MAX_LAYOUTS = 10
USE_ANCHOR_MHZ = False
SWEEP_MHZ = [10, 400]
```

### Full 10,000 layout run

```python
START_LAYOUT = 0
MAX_LAYOUTS = None
SKIP_SIMULATE = False
USE_ANCHOR_MHZ = True
ECADSTAR_WAIT_TIMEOUT_SEC = 172800  # increase if needed
```

Expect **1 impedance** + **N MHz** ECADStar batch runs (N = number of anchors, e.g. 13).

---

## Module layout

```
pipelines/dataset_sim/
  README.md                          ← this file
  run_combinations_sim_pipeline.py   ← main entry
  combinations.py                    ← load CSV
  peb.py                             ← build impedance / distribution .peb
  ecadstar.py                        ← stage PEB, run AHK, wait for PI outputs
  move_outputs.py                    ← move PI-* folders to destination
```

Related:

- `pipelines/heatmaps/sample_new_combinations.py` — create `combinations.csv`
- `scrap/orchestration/run_multifreq_sweep_pipeline.py` — VAE QC sweep (different purpose)
- `scrap/generation/generate_peb.py` — low-level PEB XML builder

---

## Requirements

- WSL Ubuntu project root: `/home/ubuntu/genai_pdn`
- Python venv: `.venv/bin/python`
- **Windows** with eCADSTAR PI/EMI, AutoHotkey v2, and paths in CONFIG
- Design `.erf` and PowerBus name matching your layout

---

## What this pipeline does **not** do

- Does not train the VAE or run model QC sweeps
- Does not normalize heatmaps or build `manifest.csv` (use existing dataset pipelines after simulation)
- Does not rename `PI-*` contents — only moves folders as ECADStar produced them
