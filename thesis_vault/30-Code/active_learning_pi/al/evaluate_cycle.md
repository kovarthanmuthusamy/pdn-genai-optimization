---
title: evaluate_cycle
type: code
path: active_learning_pi/al/evaluate_cycle.py
group: active_learning_pi/al
loc: 351
tags: [code, active_learning_pi, uncommitted]
---

# evaluate_cycle

> Full-cycle evaluation: pre/post fine-tune ECAD labels + training off-anchor metrics.

**Source:** `active_learning_pi/al/evaluate_cycle.py` · 351 lines
**Git:** uncommitted — not yet tracked

## Constants

| Name | Value |
|------|-------|
| `PRE_FINETUNE_EVAL` | `'eval_off_anchor_pre_finetune.json'` |
| `POST_FINETUNE_EVAL` | `'eval_off_anchor_post_finetune.json'` |
| `POST_FINETUNE_SCORED` | `'scored_candidates_post_finetune.json'` |
| `ACQUISITION_RANK_QUALITY` | `'acquisition_rank_quality.json'` |
| `CYCLE_SUMMARY` | `'eval_cycle_summary.json'` |
| `LEGACY_EVAL` | `'eval_off_anchor.json'` |

## Functions

- **`_off_anchor_mhz(cfg: dict)`**
- **`_eval_cfg(cfg: dict)`**
- **`evaluate_scored_vs_manifest(cfg: dict, manifest: list[dict[str, Any]], scored: list[dict[str, Any]])`**
- **`compute_acquisition_rank_quality(cfg: dict, groot: Path, iteration: int, *, manifest: list[dict[str, Any]] | None=None)`** — Rank quality from acquisition-time ``scored_candidates.json`` vs ECAD labels.
- **`save_eval_report(it_dir: Path, name: str, report: dict[str, Any])`**
- **`read_training_off_anchor_metrics(cfg: dict, groot: Path, *, epoch: int | None=None, kinds: tuple[str, ...]=('layout_cross',))`** — Latest rows from ``metrics/off_anchor_eval.csv`` for the experiment.
- **`_ingested_candidates(cfg: dict, groot: Path, iteration: int)`**
- **`cmd_infer_post_finetune(cfg: dict, groot: Path, iteration: int)`** — Re-infer ingested ECAD layouts with the current checkpoint (post fine-tune).
- **`_delta(pre: float | None, post: float | None)`**
- **`build_cycle_summary(cfg: dict, groot: Path, iteration: int, *, pre_report: dict[str, Any] | None, post_report: dict[str, Any], training_metrics: dict[str, Any] | None=None, acquisition_rank_quality: dict[str, Any] | None=None)`**
- **`cmd_post_finetune_eval(cfg: dict, groot: Path, iteration: int)`** — Re-infer ingested layouts + evaluate vs ECAD + write cycle summary.
- **`cmd_evaluate_pre_finetune(cfg: dict, groot: Path, iteration: int)`** — Evaluate acquisition-time predictions vs ingested ECAD labels.

## Imports

- [[active_learning_pi.al.paths]]
- [[candidates]]
- [[config]]
- [[decision_report]]
- [[evaluate_off_anchor]]
- [[evaluate_report]]
- [[inference_pool]]

## Imported by

- [[evaluate_report]]
- [[pipeline]]

## External dependencies

`torch`
