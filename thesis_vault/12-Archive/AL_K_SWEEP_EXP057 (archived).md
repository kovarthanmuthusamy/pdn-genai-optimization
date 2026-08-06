---
title: AL_K_SWEEP_EXP057 (archived)
type: archive
source: docs/_archive/AL_K_SWEEP_EXP057.md
tags: [archive]
---

> [!caution] Archived implementation note — not for thesis citation.

# AL Per-K Pools (K = 3–10)

### 📝 Summary of Changes

- **`active_learning_pi/al/per_k_acquire.py`**: For **each K**, generate 400 candidates, score, pick worst 10.
- **`k_config.py`**: `candidates_per_k`, `worst_per_k`, `k_sweep_per_pool`, `total_ecad_batch_size()`.
- **`exp057.json`**: `candidates_per_k: 400`, `worst_per_k: 10`, ECAD timeout 5 h for 80 PI jobs.

- **`mhz_strata.py`** + **`candidates.py`**: Stratified MHz per K pool; stratified worst-10 ECAD selection by MHz.
- **`exp057.json`**: `mhz_strata_per_k`, `worst_mhz_strata_per_k`.

### 🚀 Implementation Details

**Per K pool (400 candidates):**
- **Occupancy**: random K-hot layout (varies)
- **MHz**: fixed quotas — 80 each at 90/265/435/510, 20 each at 100/200/250/500

**Per K ECAD (10 selected):**
- Worst **2** each at 90, 265, 435, 510 MHz
- Worst **1** each at 100, 200 MHz (anchors; 250/500 omitted from ECAD quota)

**One full cycle (`COMMAND = full`):**

| K | Candidates scored | ECAD sims |
|---|-------------------|-----------|
| 3 | 400 | 10 |
| 4 | 400 | 10 |
| … | … | … |
| 10 | 400 | 10 |
| **Total** | **3200** | **80** |

Flow per K: generate → MC infer (6 passes) → worst 10 → then one combined PEB + ECAD batch.

Outputs: `k_sweep_summary.json` with per-K badness stats.

### 🛠️ Verification

```bash
python -c "
from repo_paths import setup_path; setup_path()
from active_learning_pi.al.config import load_config
from active_learning_pi.al.k_config import resolve_k_values, candidates_per_k, worst_per_k, total_ecad_batch_size
cfg = load_config('active_learning_pi/config/exp057.json')
print('K', resolve_k_values(cfg))
print('per K', candidates_per_k(cfg), 'worst', worst_per_k(cfg))
print('total ECAD', total_ecad_batch_size(cfg))
"
```

Expected: K 3–10, 400/10 per K, **80** total ECAD.

MHz strata smoke test (no GPU):

```bash
python3 -c "
from collections import Counter
from repo_paths import setup_path; setup_path()
from active_learning_pi.al.config import load_config
from active_learning_pi.al.candidates import generate_candidates
from active_learning_pi.al.mhz_strata import resolve_mhz_strata_per_k
cfg = load_config('active_learning_pi/config/exp057.json')
strata = resolve_mhz_strata_per_k(cfg)
cands = generate_candidates(num_candidates=400, mhz_grid=cfg['mhz_grid'], k_values=[5], seed=1, mhz_strata=strata)
print(dict(sorted(Counter(round(c.mhz,6) for c in cands).items())))
"
```
