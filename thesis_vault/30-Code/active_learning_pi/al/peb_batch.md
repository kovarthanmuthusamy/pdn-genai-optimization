---
title: peb_batch
type: code
path: active_learning_pi/al/peb_batch.py
group: active_learning_pi/al
loc: 41
tags: [code, active_learning_pi]
---

# peb_batch

> Build a single combined ECADSTAR .peb for all selected worst-case candidates.

**Source:** `active_learning_pi/al/peb_batch.py` · 41 lines

## Purpose

```text
Build a single combined ECADSTAR .peb for all selected worst-case candidates.

Run:
    python active_learning_pi/al/peb_batch.py
```

## Functions

- **`build_peb_from_selection(selected: list[dict[str, Any]], peb_path: Path, *, powerbus: str, heatmap_only: bool, components: str, groot: Path)`**

## Imports

- [[generate_peb]]

## Imported by

- [[pipeline]]

## External dependencies

`numpy`, `scrap`
