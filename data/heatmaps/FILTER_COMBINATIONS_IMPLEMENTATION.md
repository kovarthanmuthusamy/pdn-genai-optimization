# Filtered `all_combinations.csv` for subsampled layouts

## Context

`datasets/subsample_multifreq_inverse_k.py --execute` reduced the multifreq training set from **49,106** to **19,499** layouts (`datasets/data_multifreq_norm/manifest.csv`). Each retained layout has a matching occupancy map at `layouts/{design_id}/occ.npy` (legacy `Occ_map/` is empty after layout-store migration).

`data/heatmaps/all_combinations.csv` still listed all 49,106 decap combinations, so ECADStar batch `.peb` files were simulating deleted layouts.

## What changed (2026-06-11)

| Artifact | Before | After |
|----------|--------|-------|
| `all_combinations.csv` | 49,106 rows | **19,499 rows** |
| `combined_all.peb` | 49,106 PI-Distribution groups | **19,499 groups** |
| Backup | — | `all_combinations.csv.bak_49106_20260611T120050Z` |
| Index map | — | `decap_index_map.csv` |

### Row selection rule

A CSV row is kept when its **original row index** equals `decap_index` for a `design_id` still present in `manifest.csv`, and the row matches `layouts/{design_id}/occ.npy`.

Rows are written **sorted by original `decap_index`** (not renumbered in the manifest — `decap_index` values like `26858` are preserved in the map file).

### `decap_index_map.csv`

| Column | Meaning |
|--------|---------|
| `peb_row` | 0-based row in filtered CSV / PI-N folder after ECADStar batch (`PI-1` → row 0) |
| `original_decap_index` | Row index in the original 49,106-line `all_combinations.csv` |
| `design_id` | Layout key in `datasets/data_multifreq_norm` (e.g. `layout_pi26859_d26858`) |

Use this map when ingesting new anchor-frequency PI folders from simulation back into the dataset.

## Script

```bash
# Preview
python data/heatmaps/filter_combinations_from_manifest.py

# Apply (backs up CSV, writes map, regenerates combined_all.peb @ 63 MHz)
python data/heatmaps/filter_combinations_from_manifest.py --execute

# Custom anchor for base PEB
python data/heatmaps/filter_combinations_from_manifest.py --execute --freq-mhz 200
```

## Regenerating per-MHz PEB files

After filtering, regenerate frequency variants from the new base PEB:

```bash
# Single frequency (edit set_freq in script)
python data/heatmaps/change_frequency.py

# Or batch (example)
cd data/heatmaps
for mhz in 10 63 80 130 150 200 250 270 330 400 500; do
  python3 -c "
import re
mhz='$mhz'
c=open('combined_all.peb').read()
c,n=re.subn(r'(<EditPIDistribution Frequency=\")[^\"]*(\")', rf'\g<1>{mhz}e6\2', c)
open(f'combined_all_{mhz}MHz.peb','w').write(c)
print(f'combined_all_{mhz}MHz.peb', n)
"
done
```

Anchor list for training is in `configs/multifreq_anchors.yaml`. After simulating new MHz folders into Raw, append with:

```bash
python pipelines/data/Data_processing_multifreq.py --append --freqs-mhz 230 300 450 550 600
python scripts/Normalization.py --append
```

## Raw data note

If `C:\Users\muthusamy\Desktop\Raw\decap_combinations\all_combinations.csv` is still the full 49,106-row file, copy the filtered CSV there (or symlink) before running new multifreq simulations so PI folder order matches the `.peb` batch.
