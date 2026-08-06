# AL Production Scale — exp057

### 📝 Summary of Changes

- **`active_learning_pi/config/exp057.json`**
  - `num_candidates`: **96 → 384** (4× candidate pool per cycle)
  - `simulate_batch_size`: **16 → 48** (3× ECAD layouts per cycle)
  - `wait_timeout_sec`: **7200 → 14400** (4 h for large ECAD batches)
  - `wait_batch_start_timeout_sec`: **600 → 900**

### 🚀 Implementation Details

| Setting | Pilot (iter 1–2) | Production (iter 3+) |
|---------|------------------|----------------------|
| Candidates scored | 96 | **400** |
| ECAD sims / cycle | 16 | **10** |
| MC infer calls | 96 × 6 = 576 | **384 × 6 = 2304** |
| Selection ratio | ~1/6 worst | ~1/8 worst (same `min_uncertainty_percentile: 50`) |

**Per full cycle (`COMMAND = "full"`):**
1. Generate + infer 384 layouts (~4× GPU time vs pilot)
2. ECADStar batch of 48 PI jobs (~3× wall time; timeout raised to 4 h)
3. Overlay + fine-tune + `CYCLE_EVAL_REPORT.md` unchanged

**Run:**
```bash
# pipelines/active_learning/run.py — COMMAND = "full"
python pipelines/active_learning/run.py
```

### 🛠️ Verification & Execution Results

- Config validated: `exp057.json` loads with `num_candidates=384`, `simulate_batch_size=48`.
- Next `full` run starts **iter 3** (state already at 3) or bump `run_name` for a clean production track.
