# Fix: incomplete 430 MHz append (10k vs 29,499)

## Summary of Changes

- Diagnosed interrupted `merged_430` append (25,000 / 29,499 manifest rows; meta stale at 10,000).
- Resumed append for remaining 4,499 combinations rows → **29,499** rows at 430 MHz.
- Added `pipelines/dataset_sim/run_append_locked.py` — serializes append jobs (flock) and refreshes `dataset_meta.json` on success.
- Updated `trigger_append.py` to use the locked runner (prevents overlapping appends corrupting/interrupting each other).

## Root cause

1. **First append was killed mid-run** (~26,532 / 29,499 tasks). Manifest had 19,499 legacy + 5,501 combinations = 25,000.
2. **`dataset_meta.json` was stale** — last written by `append_restore_49k_legacy_multifreq.py` at 13:08 UTC, still showing `"430": 10000` from an older partial dataset state.
3. **Overlapping background appends** (sim pipeline fires one per MHz without waiting) can interrupt each other when writing to the same `data_multifreq_train` tree.

## Verification & Execution Results

After resume:

```text
430MHz by segment: legacy=19,499  combinations=10,000  total=29,499
dataset_meta.json samples_per_mhz["430"]: 29499
```

Re-run a single MHz if needed:

```bash
python pipelines/dataset_sim/run_append_locked.py --mhz 430 --append-tag merged_430
```

Dry-run shows remaining tasks when incomplete:

```bash
python pipelines/data/append_merged_combinations_multifreq.py --mhz 430 --append-tag merged_430 --dry-run
```

(Use `DRY_RUN = True` in script config for dry-run until `--dry-run` CLI is added.)
