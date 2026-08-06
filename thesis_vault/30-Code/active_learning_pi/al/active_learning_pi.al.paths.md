---
title: paths
type: code
path: active_learning_pi/al/paths.py
group: active_learning_pi/al
loc: 36
tags: [code, active_learning_pi]
---

# paths

> Project and run-directory path resolution for active learning.

**Source:** `active_learning_pi/al/paths.py` · 36 lines

## Purpose

```text
Project and run-directory path resolution for active learning.

Run:
    Import only — ``from active_learning_pi.al.paths import repo_root, iteration_dir``.
```

## Functions

- **`repo_root(start: Path | None=None)`** — Return repository root (delegates to ``repo_paths.REPO_ROOT``).
- **`gan_root(start: Path | None=None)`** — Deprecated alias for :func:`repo_root`.
- **`al_root(groot: Path | None=None)`**
- **`run_dir(cfg: dict, groot: Path | None=None)`**
- **`iteration_dir(cfg: dict, iteration: int, groot: Path | None=None)`**

## Imports

- [[repo_paths]]

## Imported by

- [[build_overlay]]
- [[config]]
- [[decision_report]]
- [[evaluate_cycle]]
- [[evaluate_hole_finding]]
- [[evaluate_report]]
- [[inference_pool]]
- [[per_k_acquire]]
- [[pipeline]]

## External dependencies

`repo_paths`
