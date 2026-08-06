---
title: consolidate_data_dirs
type: code
path: tools/consolidate_data_dirs.py
group: tools
loc: 143
tags: [code, tools]
---

# consolidate_data_dirs

> Move legacy data folders into data/ and remove empty legacy directories.

**Source:** `tools/consolidate_data_dirs.py` · 143 lines

## Constants

| Name | Value |
|------|-------|
| `ROOT` | `Path(__file__).resolve().parents[1]` |
| `MOVES` | `[(ROOT / 'New_heatmaps', ROOT / 'data' / 'heatmaps'), (ROOT / 'Latent_opm' / 'runs', ROOT…` |
| `ECADSTAR_FILES` | `['inspect_raw_folder.ps1']` |
| `REMOVE_DIRS` | `[ROOT / 'Data_Creation', ROOT / 'scripts', ROOT / 'New_heatmaps', ROOT / 'Latent_opm']` |
| `TEXT_REPLACEMENTS` | `[('New_heatmaps', 'data/heatmaps'), ('Latent_opm/runs', 'data/latent_runs'), ('repo_root …` |

## Functions

- **`_merge_move(src: Path, dst: Path)`**
- **`main()`**
