---
title: build_overlay
type: code
path: active_learning_pi/al/build_overlay.py
group: active_learning_pi/al
loc: 282
tags: [code, active_learning_pi]
---

# build_overlay

> Build layout_store overlay dataset from AL iterations (Option B: heatmap-only labels).

**Source:** `active_learning_pi/al/build_overlay.py` · 282 lines

## Purpose

```text
Build layout_store overlay dataset from AL iterations (Option B: heatmap-only labels).

Run:
    python -m active_learning_pi.al.build_overlay --config active_learning_pi/config/exp057.json
```

## Functions

- **`_load_imp_stats(stats_json: Path)`**
- **`_next_sample_index(heatmap_dir: Path)`**
- **`_read_manifest_rows(path: Path)`**
- **`_write_manifest_rows(path: Path, rows: list[dict[str, Any]])`**
- **`_candidate_lookup(scored_path: Path)`**
- **`collect_iteration_samples(cfg: dict, groot: Path, iteration: int)`**
- **`build_overlay_from_iterations(cfg: dict, groot: Path, *, iterations: list[int] | None=None, dry_run: bool=False)`** — Merge completed AL iterations into a layout_store overlay dataset.
- **`shutil_copy_stats(src: Path, dst: Path)`**
- **`main(argv: list[str] | None=None)`**

## Imports

- [[active_learning_pi.al.paths]]
- [[config]]
- [[multifreq_anchors]]
- [[multifreq_layout_store]]
- [[normalize_labels]]
- [[robust_normalize]]

## Imported by

- [[finetune_exp057]]
- [[finetune_exp058]]
- [[pipeline]]

## External dependencies

`numpy`, `src_vae`
