---
title: k_config
type: code
path: active_learning_pi/al/k_config.py
group: active_learning_pi/al
loc: 77
tags: [code, active_learning_pi, uncommitted]
---

# k_config

> Resolve decap-count K settings for active-learning candidate pools.

**Source:** `active_learning_pi/al/k_config.py` · 77 lines
**Git:** uncommitted — not yet tracked

## Functions

- **`resolve_k_values(cfg: dict[str, Any])`** — Return sorted K list from ``k_values``, ``k_min``/``k_max``, or ``fixed_k``.
- **`use_per_k_pools(cfg: dict[str, Any])`** — When true, each K gets its own candidate pool (e.g. 400) and worst-N selection.
- **`candidates_per_k(cfg: dict[str, Any])`**
- **`worst_per_k(cfg: dict[str, Any])`**
- **`resolve_worst_per_k_values(cfg: dict[str, Any])`** — Optional: distribute a total ECAD target across K values.
- **`total_ecad_batch_size(cfg: dict[str, Any])`**
- **`stratify_ecad_by_k(cfg: dict[str, Any])`**

## Imported by

- [[mhz_strata]]
- [[per_k_acquire]]
- [[pipeline]]
