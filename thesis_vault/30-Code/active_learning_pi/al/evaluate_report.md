---
title: evaluate_report
type: code
path: active_learning_pi/al/evaluate_report.py
group: active_learning_pi/al
loc: 460
tags: [code, active_learning_pi, uncommitted]
---

# evaluate_report

> Markdown cycle evaluation report (pre/post fine-tune + training metrics).

**Source:** `active_learning_pi/al/evaluate_report.py` · 460 lines
**Git:** uncommitted — not yet tracked

## Constants

| Name | Value |
|------|-------|
| `CYCLE_REPORT_MD` | `'CYCLE_EVAL_REPORT.md'` |

## Functions

- **`_fmt(v: Any, *, digits: int=4)`**
- **`_improvement_label(pre: float | None, post: float | None)`**
- **`read_loss_epoch_row(cfg: dict, groot: Path, epoch: int)`**
- **`collect_finetune_metrics(cfg: dict, groot: Path, *, end_epoch: int | None)`**
- **`collect_iteration_context(cfg: dict, groot: Path, iteration: int)`**
- **`_row_table(headers: list[str], rows: list[list[str]])`**
- **`_per_sample_rows(pre_rows: list[dict[str, Any]] | None, post_rows: list[dict[str, Any]] | None)`**
- **`build_cycle_report_markdown(cfg: dict, groot: Path, summary: dict[str, Any], *, finetune: dict[str, Any] | None=None, context: dict[str, Any] | None=None)`**
- **`write_cycle_report_md(cfg: dict, groot: Path, iteration: int, summary: dict[str, Any])`** — Write markdown report under iteration dir and copy pointer at run root.
- **`cmd_write_cycle_report(cfg: dict, groot: Path, iteration: int)`** — Regenerate markdown report from existing ``eval_cycle_summary.json``.

## Imports

- [[active_learning_pi.al.paths]]
- [[config]]
- [[decision_report]]
- [[evaluate_cycle]]
- [[finetune_run]]
- [[per_k_acquire]]

## Imported by

- [[evaluate_cycle]]
- [[pipeline]]

## External dependencies

`torch`
