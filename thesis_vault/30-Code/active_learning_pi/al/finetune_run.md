---
title: finetune_run
type: code
path: active_learning_pi/al/finetune_run.py
group: active_learning_pi/al
loc: 166
tags: [code, active_learning_pi]
---

# finetune_run

> Resolve AL fine-tune schedule from ``last_model.pt`` checkpoint epoch.

**Source:** `active_learning_pi/al/finetune_run.py` · 166 lines

## Functions

- **`prepare_line_buffered_logging()`** — Avoid parent prints appearing after the training subprocess when stdout is redirected.
- **`_load_yaml_like(path: Path)`**
- **`_write_yaml_like(path: Path, data: dict[str, Any])`**
- **`_default_exp_dir(cfg: dict)`**
- **`resolve_checkpoint_path(cfg: dict, groot: Path)`**
- **`checkpoint_epoch(ckpt_path: Path)`**
- **`write_runtime_finetune_config(cfg: dict, groot: Path)`** — Merge base AL finetune yaml with resume epoch + ``last_model.pt``.
- **`finetune_env(cfg: dict, groot: Path, *, use_overlay: bool)`**

## Imported by

- [[evaluate_report]]
- [[finetune_exp057]]
- [[finetune_exp058]]
- [[finetune_occ_only_exp057]]
- [[finetune_occ_only_exp058]]
- [[pipeline]]

## External dependencies

`torch`
