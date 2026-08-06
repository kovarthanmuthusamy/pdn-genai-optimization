# Dataset: layout, anchors, and sampling

Canonical description of training data for the PDN surrogate. Paths are relative to the repository root `/home/ubuntu/genai_pdn`.

---

## 1. Primary datasets

| Path | Role |
|------|------|
| `datasets/data_multifreq_train` | **Raw** layout store: `heatmap/`, `PI_freq/`, `layouts/`, `manifest.csv`, `dataset_meta.json` |
| `datasets/data_multifreq_train_norm_unbounded` | **Normalized training set** (robust log1p per-MHz, unbounded z, **K≤30**) |

### Legacy / optional

| Path | Role |
|------|------|
| `datasets/data_multifreq_train_norm_robust` | Bounded robust norm (earlier experiments) |
| `data_multi_norm_robust` | Older backup at repo root |

### Layout store convention

Real files only (no training-critical symlinks):

```text
layouts/{design_id}/occ.npy   # occupancy (52,)
layouts/{design_id}/imp.npy   # impedance spectrum (231,)
```

Legacy 19,499 layouts and later 10,000 combinations were merged into one train pool (see §4).

---

## 2. `dataset_meta.json`

Each processing or normalization run writes a JSON summary at the dataset root via `libs/dataset_meta.py`.

**Written by:** `pipelines/data/processing_*.py`, `pipelines/normalize/multifreq.py`, `pipelines/normalize/apply_stats.py`.

Typical fields: `schema_version`, `stage` (`raw` | `normalized`), counts (manifest rows, unique layouts, heatmap/PI files), size, `pi_frequencies_mhz`, `samples_per_mhz`. Normalized sets may include a `normalization` block and `max_k_filter`.

### Anchor resolution order

`load_anchors_mhz(dataset_dir)` resolves training anchors from:

1. `{dataset_dir}/dataset_meta.json` → `pi_frequencies_mhz`
2. `{dataset_dir}/manifest.csv` (`freq_mhz`)
3. `datasets/data_multifreq_train/dataset_meta.json`
4. Legacy YAML / defaults

After appending new MHz bands, refresh raw meta so `pi_frequencies_mhz` matches the manifest, then re-normalize if per-MHz stats must change.

---

## 3. K≤30 filter (training scope)

Practical PDN budgets focus on moderate decap counts. Normalization build `build_train_norm_unbounded.py` sets `MAX_K = 30`:

- \(K\) per `design_id` from `(occ > 0.5).sum()`
- Layouts with \(K > 30\) dropped before stats and output
- Stats (median/IQR per MHz, impedance log z-score) fit **only** on the kept subset

Verified post-build example (unbounded set):

- ~24,979 layouts kept, K∈[0,30]
- Manifest filtered accordingly; `normalization_stats.json` records `"max_k_filter": 30`

**Pros:** Matches design budgets of interest; reduces extreme high-\(K\) dominance.  
**Cons:** Surrogate not trained for \(K>30\); claims must stay within this scope.

---

## 4. Layout combination sampling and merge

### Sources

| File | Rows | Role |
|------|------|------|
| `data/heatmaps/all_combinations.csv` | 19,499 | Legacy inverse-K subsample |
| `data/heatmaps/combinations.csv` | 10,000 | Later sample (no overlap with legacy) |
| `data/heatmaps/all_combinations_merged.csv` | 29,499 | Unified list for PEB generation |

### New sample policy (`pipelines/heatmaps/sample_new_combinations.py`)

- **K=2:** all remaining unique pairs not already in the legacy CSV
- **K=3…50:** inverse-K allocation to fill `TOTAL_N` (default 10,000)

Inverse-K puts more mass on small \(K\) (combinatorially denser, more design-relevant).

### ECADStar PI numbering

CSV row \(i\) maps to **PI-\((i+1)\)** in PEB script order. Merged order keeps legacy rows first, then new combinations. Use index maps (`decap_index_map.csv`, `merged_combinations_index_map.csv`) for `design_id` — do not assume contiguous legacy `decap_index` values.

### Train pool composition (illustrative)

| `source_folder` | Layouts | Anchors |
|-----------------|---------|---------|
| `layout` / `peb_batch` | 19,499 | Full 16-MHz set (base + appended bands) |
| `combinations` | 10,000 | Same 16 MHz in combinations sim batches |

Exact MHz list is authoritative in `dataset_meta.json`, not in outdated YAML alone.

---

## 5. Spectrum diversity analysis

`pipelines/analysis/impedance_decade_diversity.py` ranks the **231** spectrum bins by cross-layout diversity (e.g. `std_log10` of \(|Z|\)). Outputs: `experiments/impedance_freq_diversity.json` / `.npz`.

Use: motivate which frequency regions are information-rich for acquisition or evaluation — not a training transform by itself.

---

## 6. Maintenance commands

```bash
python pipelines/dataset/clean_dataset_symlinks.py --execute
python pipelines/dataset/clean_train_metadata.py --execute
python pipelines/normalize/build_train_norm_unbounded.py
```

Raw root should keep `manifest.csv` and `dataset_meta.json`; transient `append_*` / progress backups are cleanup targets.

---

## 7. Scale notes (for thesis context)

- Full discrete space: \(2^{52}\) placements × continuous frequency — intractable.
- Training uses a **subsampled** layout pool (tens of thousands of designs) × discrete MHz anchors.
- Off-anchor generalization (between anchors) is both a **data density** and a **model** issue; AL targets the latter with selective new sims ([active-learning.md](./active-learning.md)).
