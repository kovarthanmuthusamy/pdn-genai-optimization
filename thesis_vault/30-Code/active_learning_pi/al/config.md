---
title: config
type: code
path: active_learning_pi/al/config.py
group: active_learning_pi/al
loc: 52
tags: [code, active_learning_pi]
---

# config

> Load/save active-learning pipeline configuration (JSON or YAML).

**Source:** `active_learning_pi/al/config.py` · 52 lines

## Purpose

```text
Load/save active-learning pipeline configuration (JSON or YAML).

Run:
    Import only — path set via ``CONFIG_PATH`` in ``pipelines/active_learning/run.py``.
```

## Functions

- **`load_config(path: str | Path | None=None)`**
- **`save_json(path: Path, obj: Any)`**
- **`load_json(path: Path)`**

## Imports

- [[active_learning_pi.al.paths]]

## Imported by

- [[build_overlay]]
- [[decision_report]]
- [[evaluate_cycle]]
- [[evaluate_hole_finding]]
- [[evaluate_report]]
- [[finetune_exp057]]
- [[finetune_exp058]]
- [[finetune_occ_only_exp057]]
- [[finetune_occ_only_exp058]]
- [[per_k_acquire]]
- [[pipeline]]
- [[validate_acquisition_ab]]

## External dependencies

`yaml`
