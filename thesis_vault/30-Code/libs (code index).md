---
title: libs (code index)
type: index
tags: [index, code, libs]
---

# libs/ — code index

10 modules.

## `libs/`

- [[libs.__init__]]
- [[dataset_meta]] — Write ``dataset_meta.json`` summarizing an on-disk training dataset.
- [[experiment_paths]] — Shared experiment config loading with repo-relative path resolution.

## `libs/data_creation/`

- [[libs.data_creation.__init__]] — Package: libs.data_creation
- [[csv_to_occupancy]] — 52-d decap vector → binary occupancy vector conversion utilities.
- [[libs.data_creation.heatmap]] — Parse ECADStar .map files and interpolate PI-Distribution heatmaps (64×64).
- [[libs.data_creation.impedance]] — Read IC1 impedance CSV files and plot log-log impedance profiles.
- [[occupancy]] — Map 52-d decap vectors to 7×8 physical occupancy grids.

## `libs/peb/`

- [[libs.peb.__init__]] — Package: libs.peb
- [[frequency]] — PEB PI-Distribution frequency replacement (shared by change_frequency / regenerate_mhz_pebs).
