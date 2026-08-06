---
title: mhz_strata
type: code
path: active_learning_pi/al/mhz_strata.py
group: active_learning_pi/al
loc: 60
tags: [code, active_learning_pi, uncommitted]
---

# mhz_strata

> Stratified MHz quotas for per-K AL candidate pools and ECAD selection.

**Source:** `active_learning_pi/al/mhz_strata.py` · 60 lines
**Git:** uncommitted — not yet tracked

## Functions

- **`_parse_strata(raw: dict[str, Any] | dict[float, int])`**
- **`resolve_mhz_strata_per_k(cfg: dict[str, Any])`**
- **`resolve_worst_mhz_strata_per_k(cfg: dict[str, Any])`** — ECAD MHz quotas per K; auto-scaled from pool strata if omitted.
- **`scale_worst_mhz_strata_from_pool(pool: dict[float, int], n_worst: int)`** — Scale pool MHz strata to sum to ``n_worst`` using largest remainder.

## Imports

- [[k_config]]

## Imported by

- [[per_k_acquire]]
