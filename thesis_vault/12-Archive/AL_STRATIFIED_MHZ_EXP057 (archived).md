---
title: AL_STRATIFIED_MHZ_EXP057 (archived)
type: archive
source: docs/_archive/AL_STRATIFIED_MHZ_EXP057.md
tags: [archive]
---

> [!caution] Archived implementation note — not for thesis citation.

# AL Stratified MHz (exp057)

### 📝 Summary of Changes

- **`active_learning_pi/al/mhz_strata.py`**: Resolve `mhz_strata_per_k` and `worst_mhz_strata_per_k` from config (auto-scale if omitted).
- **`active_learning_pi/al/candidates.py`**: `mhz_strata` + `build_mhz_schedule()` for fixed MHz counts per pool.
- **`active_learning_pi/al/acquisition.py`**: `mhz_quotas` → stratified worst selection by MHz.
- **`active_learning_pi/al/per_k_acquire.py`**: Wire strata for generate + ECAD; log distributions in `k_sweep_summary.json`.
- **`active_learning_pi/al/evaluate_report.py`**: Show MHz strata and ECAD MHz distribution in `CYCLE_EVAL_REPORT.md`.
- **`active_learning_pi/config/exp057.json`**: Strata quotas for pool and ECAD.

### 🚀 Implementation Details

**Per K (occupancy varies, MHz stratified):**

| MHz | Pool count | ECAD worst |
|-----|------------|------------|
| 90, 265, 435, 510 | 80 each | 2 each |
| 100, 200, 250, 500 | 20 each | 1 each (100, 200 only) |

- **3200** candidates scored per cycle (8×400).
- **80** ECAD sims per cycle (8×10), biased toward off-anchor MHz.

Legacy mode (`k_sweep_per_pool: false`) ignores strata unless `mhz_strata` is passed explicitly.

### 🔎 Exploration MHz (cheap discovery)

To let uncertainty “discover” weak frequency regions (e.g. near **270 MHz**) without paying for a labeled sweep,
you can reserve a small slice of each per-K pool for **exploration frequencies**. These candidates still use MC
inference/uncertainty (cheap, unlabeled) and only the selected worst points get ECAD labels.

Config keys (optional):

- `explore_candidates_per_k` (int): number of candidates per K reserved for exploration.
- `explore_mhz_grid` (list[float] | null): MHz values to sample exploration from (uniform).
- `explore_mhz_min` / `explore_mhz_max` (float): range for uniform exploration when grid is null.
- `explore_mhz_quantize` (float): quantize exploration MHz (e.g. 5 or 10) to limit unique values.
- `explore_mhz_band_edges` (list[float] | null): if set, draws uniformly across bands defined by edges.

When `mhz_strata_per_k` is set, the main strata are **automatically scaled** to `candidates_per_k - explore_candidates_per_k`,
then the exploration slice is shuffled into the pool.

### 🛠️ Verification & Execution Results

```text
pool strata sum 400 {90: 80, 265: 80, 435: 80, 510: 80, 100: 20, 200: 20, 250: 20, 500: 20}
worst mhz sum 10 {90: 2, 265: 2, 435: 2, 510: 2, 100: 1, 200: 1}
candidate MHz dist matches quotas
selected MHz {90: 2, 100: 1, 200: 1, 265: 2, 435: 2, 510: 2}
```

Status: **verified** (CPU smoke test, no GPU infer).
