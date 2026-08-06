---
title: per_k_acquire
type: code
path: active_learning_pi/al/per_k_acquire.py
group: active_learning_pi/al
loc: 198
tags: [code, active_learning_pi, uncommitted]
---

# per_k_acquire

> Per-K candidate pools: N candidates and worst-M selection for each K value.

**Source:** `active_learning_pi/al/per_k_acquire.py` · 198 lines
**Git:** uncommitted — not yet tracked

## Constants

| Name | Value |
|------|-------|
| `SELECTED_JSON` | `'selected_for_simulation.json'` |
| `K_SWEEP_SUMMARY` | `'k_sweep_summary.json'` |

## Functions

- **`_write_selected_csv(it_dir: Path, selected: list[dict[str, Any]])`**
- **`run_per_k_acquire(cfg: dict, groot: Path, iteration: int)`** — For each K in ``k_min..k_max``:
- **`load_k_sweep_summary(cfg: dict, iteration: int, groot: Path)`**

## Imports

- [[acquisition]]
- [[active_learning_pi.al.paths]]
- [[candidates]]
- [[config]]
- [[inference_pool]]
- [[k_config]]
- [[mhz_strata]]

## Imported by

- [[evaluate_report]]
- [[pipeline]]
