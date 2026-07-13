# Sample new layout combinations (10k inverse-K)

### 📝 Summary of Changes

- **K=2:** every remaining unique pair (all C(52,2) not in `all_combinations.csv`).
- **K=3..50:** inverse-K allocation for the rest of `TOTAL_N` (default 10,000).

### 🚀 Implementation Details

**Run:**

```bash
.venv/bin/python pipelines/heatmaps/sample_new_combinations.py
```

**Config (top of script):**

| Key | Default | Role |
|-----|---------|------|
| `EXISTING_CSV` | `data/heatmaps/all_combinations.csv` | Layouts to exclude |
| `OUTPUT_CSV` | `data/heatmaps/combinations.csv` | New simulation list |
| `TOTAL_N` | `10000` | Target layout count |
| `CANDIDATE_POOL_CSV` | `None` | Optional larger pool; else random K-hot generation |

**Inverse-K:** K=2 takes **all** remaining pairs (326 with current old CSV). K=3..50 share the remaining quota (~9,674) via the same inverse-exponential schedule as `subsample_inverse_k.py` (anchor at K=3).

**Outputs:**

- `data/heatmaps/combinations.csv` — 10,000 rows, no header, comma-separated 0/1
- `data/heatmaps/combinations_sample_report.json` — per-K targets and actual counts

### 🛠️ Verification & Execution Results

```
K=2: all remaining pairs → 326 layouts
K=3..50: inverse-K → 9,674 layouts (TOTAL_N=10,000)
Wrote combinations.csv (10,000 rows)
```

No overlap with `all_combinations.csv` (verified in script).
