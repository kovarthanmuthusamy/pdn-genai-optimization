---
title: combinations_multitype
type: code
path: pipelines/dataset_sim/combinations_multitype.py
group: pipelines/dataset_sim
loc: 36
tags: [code, pipelines, uncommitted]
---

# combinations_multitype

> Load multi-type decap combination rows from CSV (type codes 0/1/2).

**Source:** `pipelines/dataset_sim/combinations_multitype.py` · 36 lines
**Git:** uncommitted — not yet tracked

## Constants

| Name | Value |
|------|-------|
| `N_DECAPS` | `52` |
| `VALID_CODES` | `{0, 1, 2}` |

## Functions

- **`load_multitype_csv(path: Path)`** — Return (N, 52) int8 rows of type codes in ``VALID_CODES``.
- **`layout_dir_name(index: int)`**

## Imported by

- [[peb_multitype]]
- [[run_multitype_sim_pipeline]]

## External dependencies

`numpy`
