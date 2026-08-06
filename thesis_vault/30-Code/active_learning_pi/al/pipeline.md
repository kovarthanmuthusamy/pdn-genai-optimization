---
title: pipeline
type: code
path: active_learning_pi/al/pipeline.py
group: active_learning_pi/al
loc: 519
tags: [code, active_learning_pi]
---

# pipeline

> Active-learning orchestrator — full generate→simulate→ingest→normalize loop.

**Source:** `active_learning_pi/al/pipeline.py` · 519 lines

## Purpose

```text
Active-learning orchestrator — full generate→simulate→ingest→normalize loop.

Run:
    Prefer ``python pipelines/active_learning/run.py`` (CONFIG at top).
```

## Constants

| Name | Value |
|------|-------|
| `SELECTED_JSON` | `'selected_for_simulation.json'` |
| `PEB_NAME` | `'batch_simulate_once.peb'` |

## Functions

- **`_state_path(cfg: dict, groot: Path)`**
- **`current_iteration(cfg: dict, groot: Path)`**
- **`bump_iteration(cfg: dict, groot: Path)`**
- **`_peb_path(it_dir: Path)`**
- **`cmd_generate(cfg: dict, groot: Path, iteration: int)`**
- **`cmd_infer(cfg: dict, groot: Path, iteration: int)`**
- **`cmd_select_bad(cfg: dict, groot: Path, iteration: int)`**
- **`cmd_build_peb(cfg: dict, groot: Path, iteration: int)`**
- **`cmd_simulate(cfg: dict, groot: Path, iteration: int)`**
- **`cmd_ingest(cfg: dict, groot: Path, iteration: int)`**
- **`cmd_normalize(cfg: dict, groot: Path, iteration: int)`**
- **`cmd_evaluate(cfg: dict, groot: Path, iteration: int)`**
- **`cmd_build_overlay(cfg: dict, groot: Path)`**
- **`cmd_finetune(cfg: dict, groot: Path)`**
- **`_resolve_occ_warmup_script(cfg: dict, groot: Path)`** — Prefer exp-specific warmup script; fall back to exp058 then exp057.
- **`cmd_occ_only_warmup(cfg: dict, groot: Path)`** — Pre-AL occupancy-only fine-tune (no overlay) — run before candidate scoring.
- **`cmd_finetune_hint(cfg: dict, groot: Path)`**
- **`run_cycle(cfg: dict, groot: Path, *, do_simulate: bool, do_ingest: bool, run_finetune: bool=False, run_post_eval: bool | None=None)`** — Active-learning cycle:
- **`run_full_cycle(cfg: dict, groot: Path, *, skip_simulate: bool=False, skip_ingest: bool=False)`** — End-to-end: cycle + overlay + fine-tune (single command).
- **`main_from_config(*, command: str, config_path: str | Path | None=None, iteration: int | None=None, skip_simulate: bool=False, skip_ingest: bool=False)`** — Run pipeline from CONFIG. Called by pipelines/active_learning/run.py.

## Imports

- [[acquisition]]
- [[active_learning_pi.al.ecadstar]]
- [[active_learning_pi.al.paths]]
- [[build_overlay]]
- [[candidates]]
- [[config]]
- [[decision_report]]
- [[evaluate_cycle]]
- [[evaluate_off_anchor]]
- [[evaluate_report]]
- [[finetune_run]]
- [[inference_pool]]
- [[ingest_labels]]
- [[k_config]]
- [[multifreq_layout_store]]
- [[normalize_labels]]
- [[peb_batch]]
- [[per_k_acquire]]

## Imported by

- [[run]]

## External dependencies

`src_vae`
