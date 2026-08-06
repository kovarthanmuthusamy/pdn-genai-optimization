---
title: trigger_append
type: code
path: pipelines/dataset_sim/trigger_append.py
group: pipelines/dataset_sim
loc: 49
tags: [code, pipelines]
---

# trigger_append

> Enqueue append jobs and ensure a detached worker is running.

**Source:** `pipelines/dataset_sim/trigger_append.py` · 49 lines

## Functions

- **`append_tag_for_mhz(mhz: float)`**
- **`trigger_append_after_move(mhz: float, *, raw_root_win: str=RAW_ROOT_WIN)`** — Validate moved heatmaps, enqueue append, start worker if needed (no sim wait).

## Imports

- [[append_queue]]
- [[pipelines.dataset_sim.paths]]
- [[repo_paths]]

## Imported by

- [[run_append_worker]]
- [[run_combinations_sim_pipeline]]
- [[run_multitype_sim_pipeline]]

## External dependencies

`repo_paths`
