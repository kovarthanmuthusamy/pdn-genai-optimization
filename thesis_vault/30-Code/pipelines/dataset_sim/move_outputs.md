---
title: move_outputs
type: code
path: pipelines/dataset_sim/move_outputs.py
group: pipelines/dataset_sim
loc: 69
tags: [code, pipelines]
---

# move_outputs

> Move ECADStar PI-* folders into destination (names unchanged).

**Source:** `pipelines/dataset_sim/move_outputs.py` · 69 lines

## Functions

- **`heatmaps_dir_name(mhz: float)`** — PI-Distribution destination folder, e.g. ``heatmaps_10MHz``.
- **`_find_pi_folder(source: Path, pi_num: int)`**
- **`_clear_pi_children(dest_dir: Path)`** — Remove existing PI-* entries under ``dest_dir`` before a full move.
- **`move_pi_outputs(*, source_emc_dir: str, dest_dir: Path, pi_count: int, clean_dest: bool)`** — Move PI-1..PI-{pi_count} from EMC → ``dest_dir`` (same folder names).

## Imports

- [[pipelines.dataset_sim.ecadstar]]

## Imported by

- [[append_merged_combinations_multifreq]]
- [[pipelines.dataset_sim.paths]]
- [[run_combinations_sim_pipeline]]
- [[run_multitype_sim_pipeline]]
- [[verify_pipeline_paths]]
