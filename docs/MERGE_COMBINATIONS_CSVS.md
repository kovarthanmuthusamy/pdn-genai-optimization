# Merge decap combination CSVs for multifreq heatmap generation

### 📝 Summary of Changes

- Documented how **legacy** (`all_combinations.csv`) and **new** (`combinations.csv`) layout lists relate to PI numbering, manifest rows, and append scripts.
- Added `pipelines/heatmaps/merge_combination_csvs.py` to produce a unified 29,499-row CSV plus `merged_combinations_index_map.csv`.
- Verified alignment with `datasets/data_multifreq_train` (19,499 `layout` + 10,000 `combinations` layouts, 16 MHz each).

### 🚀 Implementation Details

#### File roles

| File | Rows | Role |
|------|------|------|
| `data/heatmaps/all_combinations.csv` | 19,499 | Legacy layouts (inverse-K subsample). Row `i` = **peb_row** `i` in `combined_all.peb`. |
| `data/heatmaps/decap_index_map.csv` | 19,499 | Maps peb_row → `original_decap_index` + `design_id` (`layout_pi*_d*`). Required for legacy append. |
| `data/heatmaps/combinations.csv` | 10,000 | New layouts (no overlap with old). Row `i` → `PI-(i+1)` in combinations sim. |
| `data/heatmaps/all_combinations_merged.csv` | 29,499 | **Old first, then new** — for unified PEB generation. |
| `data/heatmaps/merged_combinations_index_map.csv` | 29,499 | Per-row segment, `pi_number`, `design_id` for both halves. |

#### PI numbering (critical)

ECADStar names outputs in **PEB script order**:

```
CSV row i  →  PI-(i+1)
```

**Merged order:**

```
rows 0 .. 19,498      →  PI-1 .. PI-19499     (legacy, same as all_combinations.csv)
rows 19,499 .. 29,498 →  PI-19500 .. PI-29499 (combinations.csv rows 0..9999)
```

Legacy `decap_index` values are **not** contiguous (e.g. 53..49048); always use `decap_index_map.csv` or `merged_combinations_index_map.csv` for `design_id`, not raw row index.

#### Current dataset (`datasets/data_multifreq_train`)

| `source_folder` | Layouts | MHz anchors |
|-----------------|---------|-------------|
| `layout` | 19,499 | 10, 63, 80, 200, 230, 250, 270, 300, 330, 400, 500 |
| `peb_batch` | 19,499 | 350, 370, 390, 420, 450 (appended via `append_peb_batch_raw.py`) |
| `combinations` | 10,000 | All 16 MHz in one combinations sim batch |

Together, every layout in train has **16 anchors**. `configs/multifreq_anchors.yaml` still lists 11 — update it when adding new simulated MHz.

#### Generating **new anchor frequencies**

**Recommended: two simulation passes** (matches existing append tooling).

1. **Legacy 19,499 layouts** — use filtered `all_combinations.csv` + `decap_index_map.csv`:
   - Build PEB: `filter_combinations.py` / `regenerate_mhz_pebs.py` / `combined_all_{MHz}MHz.peb`
   - Simulate ECADStar → Raw `heatmap_{MHz}/PI-1..PI-19499`
   - Append: `pipelines/data/append_peb_batch_raw.py` with `ONLY_FREQS_MHZ = [<new MHz>]`

2. **New 10,000 layouts** — use `combinations.csv` only:
   - Simulate: `pipelines/dataset_sim/run_combinations_sim_pipeline.py` (`SWEEP_MHZ` or `USE_ANCHOR_MHZ=True`)
   - Append: `pipelines/data/append_combinations_multifreq.py`

3. **Normalize** after append: `pipelines/normalize/multifreq.py` (`APPEND=True`)

**Alternative: one merged PEB** (29,499 layouts):

```bash
python pipelines/heatmaps/merge_combination_csvs.py
# → all_combinations_merged.csv + merged_combinations_index_map.csv
```

Build distribution PEB from merged occupancy; simulate once per MHz. Appending:

```bash
python pipelines/heatmaps/generate_peb_from_csv.py   # CSV → .peb (merged 29k)
python pipelines/data/append_merged_combinations_multifreq.py
```

See `docs/APPEND_MERGED_COMBINATIONS_MULTIFREQ.md`.

#### Merge script

```bash
python pipelines/heatmaps/merge_combination_csvs.py
```

Outputs:

- `data/heatmaps/all_combinations_merged.csv`
- `data/heatmaps/merged_combinations_index_map.csv`
- `data/heatmaps/merged_combinations_report.json`

Set `EXECUTE = False` in the script for validation-only dry run.

### 🛠️ Verification & Execution Results

```
old 19499 unique | new 10000 unique | overlap 0
combinations manifest: row 0 / 9999 match combinations.csv occ.npy
legacy peb_row 0 / 19498 match all_combinations.csv via decap_index_map

Merge script:
  Wrote all_combinations_merged.csv (29,499 rows)
  PI-1..PI-19499 (legacy) + PI-19500..PI-29499 (combinations)
```
