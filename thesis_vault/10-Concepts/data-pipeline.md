---
title: data-pipeline
type: concept
source: docs/data-pipeline.md
tags: [concept, thesis]
---

> [!info] Mirror of `docs/data-pipeline.md` — edit the source file, then re-run `tools/build_vault.py`.

# Data and simulation pipeline

Durable description of how layout combinations become training rows. Operational bug-fix notes are archived; only stable procedures remain here.

---

## 1. End-to-end flow

```text
Sample / merge combination CSVs
  → Build PEB (ECADStar script order)
  → Simulate impedance + PI-distribution heatmaps (Windows, headless engineer.exe --batch)
  → Append raw multifreq dataset (manifest + layout store)
  → Normalize (robust log1p per-MHz; optional K≤30)
  → Train / evaluate / active-learn
```

Entry scripts generally use a **CONFIG block** at the top of the file (edit constants, then `python path/to/script.py`). Canonical runners live under `pipelines/`; see `pipelines/README.md`.

---

## 2. Combination lists and PEB order

1. Sample new layouts (`pipelines/heatmaps/sample_new_combinations.py`) or use legacy lists.
2. Merge legacy + new CSV for unified PEB generation (`merge_combination_csvs.py`).
3. **PI numbering:** CSV row \(i\) → **PI-\((i+1)\)** in ECADStar output order.

Details and file roles: [[dataset|dataset.md]] §4.

---

## 3. Multifreq simulation sweeps

Orchestration under `scrap/orchestration/` and `pipelines/dataset_sim/`:

- Generate predicted heatmaps / PEB entries for selected \(K\) and MHz grids.
- Run ECADStar batch; move/compare real vs generated maps.
- **Multi-K:** `K_VALUE` may be a scalar or list (e.g. `[10, 20, 30]`); outputs nest as `freq_<MHz>/K<k>/`.

PEB nested order (must match compare tooling):

```text
for each MHz:
  for each K:
    for each sample:
      one PI entry
```

Primary QC metrics for compare reports: [[evaluation-metrics|evaluation-metrics.md]].

---

## 4. Append into the train set

After a successful sim batch:

1. Append raw rows (CLI append helpers under `pipelines/data/`, tags per MHz batch).
2. Refresh `dataset_meta.json` so `pi_frequencies_mhz` matches the manifest.
3. Normalize with append or full rebuild (`pipelines/normalize/…`). Prefer **frozen stats** for small appends when old/new rows must stay comparable; **rebuild stats** when the MHz set or population changes materially.

Incomplete appends leave mismatched meta/counts — verify `manifest.csv` row counts vs unique layouts before training.

---

## 5. ECADStar operational notes

Simulation uses the native **headless CLI** — `engineer.exe <design.erf> --batch
<file.peb> --batch-auto-exit` — which opens the design, runs the batch, writes
`PI-1..N`, and quits itself (see `ecadstar_headless_cli.md`). No AutoHotkey, no GUI
focus, no RDP-foreground requirement. Practical rules:

- Runs in the background; the RDP/console session does **not** need to be visible or
  focused (the old AutoHotkey foreground constraint is gone).
- `engineer.exe` releases its own design lock on exit, so there is no `.rlk` churn
  between phases.
- Resume from progress files to skip completed MHz bands.

These are environment constraints, not model issues — document them in the thesis experimental setup if sims were remote-automated.

---

## 6. Path resolution

Use `repo_paths.py` as the single source of truth for repository roots and dataset directories (`REPO_ROOT`, `setup_path()`, `repo_path(...)`). Avoid hard-coded absolute paths in new scripts.

---

## 7. Related package docs

| Path | Content |
|------|---------|
| `pipelines/README.md` | Pipeline index |
| `pipelines/dataset_sim/README.md` | Combinations sim pipeline |
| [[dataset|dataset.md]] | Dataset layout and anchors |
| [[active-learning|active-learning.md]] | Selective sim inside AL cycles |


## Implemented by

- [[processing_multifreq]] — `pipelines/data/processing_multifreq.py`
- [[ingest_labels]] — `active_learning_pi/al/ingest_labels.py`
- [[active_learning_pi.al.ecadstar]] — `active_learning_pi/al/ecadstar.py`
- [[peb_batch]] — `active_learning_pi/al/peb_batch.py`
