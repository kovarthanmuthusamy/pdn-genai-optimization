---
title: peb
type: code
path: pipelines/dataset_sim/peb.py
group: pipelines/dataset_sim
loc: 80
tags: [code, pipelines]
---

# peb

> Build ECADStar .peb files for combinations batch simulation.

**Source:** `pipelines/dataset_sim/peb.py` · 80 lines

## Functions

- **`_import_generate_peb(repo_root: Path)`**
- **`mhz_to_freq_hz(mhz: float)`**
- **`build_impedance_peb(occupancy: np.ndarray, output_path: Path, *, repo_root: Path, powerbus: str, components: str)`** — CreatePISpectrum only — one PI per layout row.
- **`build_distribution_peb(occupancy: np.ndarray, output_path: Path, *, repo_root: Path, mhz: float, powerbus: str)`** — PI-Distribution only — one PI per layout row at ``mhz``.

## Imports

- [[combinations]]
- [[generate_peb]]

## Imported by

- [[generate_peb_from_csv]]
- [[run_combinations_sim_pipeline]]

## External dependencies

`numpy`, `scrap`
