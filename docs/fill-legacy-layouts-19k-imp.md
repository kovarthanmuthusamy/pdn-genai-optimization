# Fill legacy layouts from Dataset_19k impedance

## Summary of Changes

- Added `pipelines/heatmaps/fill_legacy_layouts_19k_imp.py` to materialize `imp.npy` + `occ.npy` for 19,499 legacy layouts.

## Implementation Details

**Mapping** (same as `append_restore_49k_legacy_multifreq.py`):

- `decap_index_map.csv` → `design_id`, `original_decap_index`
- Impedance PI folder: `PI-(original_decap_index + 1)` under `Dataset_19k/imp`
- Decap vector: `all_combinations.csv` row `peb_row`

**Preprocessing:** `processing_multifreq._write_layout_once` (IC1 CSV → `imp.npy`, decap → `occ.npy`).

```bash
python pipelines/heatmaps/fill_legacy_layouts_19k_imp.py              # dry-run
python pipelines/heatmaps/fill_legacy_layouts_19k_imp.py --execute --force
python pipelines/normalize/build_train_norm_unbounded.py
```

## Verification & Execution Results

Dry-run: 19,499 writable, 0 missing impedance.

Execute: written=19,499, legacy layouts with real imp/occ in `data_multifreq_train/layouts/`.
