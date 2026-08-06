---
title: peb_multitype
type: code
path: pipelines/dataset_sim/peb_multitype.py
group: pipelines/dataset_sim
loc: 88
tags: [code, pipelines, uncommitted]
---

# peb_multitype

> Build ECADStar .peb files for multi-type combinations batch simulation.

**Source:** `pipelines/dataset_sim/peb_multitype.py` · 88 lines
**Git:** uncommitted — not yet tracked

## Purpose

```text
Build ECADStar .peb files for multi-type combinations batch simulation.

Mirrors ``pipelines/dataset_sim/peb.py`` but each slot carries a type code
(0 empty / 1 type-1 / 2 type-2) and the PEB sets per-component C / ESR / ESL
from the type catalog (see ``scrap/generation/generate_peb_multitype.py``).
```

## Functions

- **`_import_generate(repo_root: Path)`**
- **`mhz_to_freq_hz(mhz: float)`**
- **`build_impedance_peb_multitype(type_codes: np.ndarray, output_path: Path, *, repo_root: Path, powerbus: str, components: str)`** — CreatePISpectrum only — one PI per layout row (type-aware).
- **`build_distribution_peb_multitype(type_codes: np.ndarray, output_path: Path, *, repo_root: Path, mhz: float, powerbus: str)`** — PI-Distribution only — one PI per layout row at ``mhz`` (type-aware).

## Imports

- [[combinations_multitype]]
- [[generate_peb_multitype]]

## Imported by

- [[run_multitype_sim_pipeline]]

## External dependencies

`numpy`, `scrap`
