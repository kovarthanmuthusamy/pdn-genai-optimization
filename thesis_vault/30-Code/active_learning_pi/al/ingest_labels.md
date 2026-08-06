---
title: ingest_labels
type: code
path: active_learning_pi/al/ingest_labels.py
group: active_learning_pi/al
loc: 117
tags: [code, active_learning_pi]
---

# ingest_labels

> Ingest ECADSTAR PI-Distribution outputs into per-sample label directories.

**Source:** `active_learning_pi/al/ingest_labels.py` · 117 lines

## Purpose

```text
Ingest ECADSTAR PI-Distribution outputs into per-sample label directories.

Run:
    python active_learning_pi/al/ingest_labels.py
```

## Functions

- **`_find_pi_folders(emc_dir: Path)`** — Map PI number → folder under EMC (PI-1, PI-2, …).
- **`ingest_simulation_outputs(cfg: dict, selected: list[dict[str, Any]], labels_dir: Path, groot: Path)`** — Read PI-1..PI-N folders from ECADStar .emc directory; extract .map per candidate MHz.

## Imports

- [[libs.data_creation.heatmap]]
- [[pipelines.dataset_sim.ecadstar]]
- [[pipelines.dataset_sim.paths]]

## Imported by

- [[pipeline]]

## External dependencies

`libs`, `numpy`, `pipelines`
