---
title: ecadstar
type: code
path: active_learning_pi/al/ecadstar.py
group: active_learning_pi/al
loc: 95
tags: [code, active_learning_pi]
---

# ecadstar

> ECADSTAR batch simulation bridge (native headless CLI).

**Source:** `active_learning_pi/al/ecadstar.py` · 95 lines

## Purpose

```text
ECADSTAR batch simulation bridge (native headless CLI).

Runs ``engineer.exe --batch --batch-auto-exit`` (no AutoHotkey / no GUI focus; see
``docs/ecadstar_headless_cli.md``): opens the .erf, runs the .peb, writes PI-1..N,
and quits itself.

Uses the same path resolution and wait logic as ``pipelines/dataset_sim/ecadstar.py``.
```

## Constants

| Name | Value |
|------|-------|
| `DEFAULT_ENGINEER_EXE` | `'C:\\Program Files\\eCADSTAR\\eCADSTAR 2023.0\\Analysis\\bin\\engineer.exe'` |

## Functions

- **`run_ecadstar_batch(cfg: dict, peb_path: Path, groot: Path, *, pi_count: int=1)`** — Stage PEB → run ECADStar headless batch → optionally wait for PI outputs.

## Imports

- [[pipelines.dataset_sim.ecadstar]]

## Imported by

- [[pipeline]]

## External dependencies

`pipelines`
