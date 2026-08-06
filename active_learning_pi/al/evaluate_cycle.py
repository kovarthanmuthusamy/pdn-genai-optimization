"""Full-cycle evaluation: pre/post fine-tune ECAD labels + training off-anchor metrics."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch

from active_learning_pi.al.config import load_json, save_json
from active_learning_pi.al.evaluate_off_anchor import (
    evaluate_acquisition_rank_quality,
    evaluate_labels_vs_predictions,
)
from active_learning_pi.al.decision_report import write_decision_report
from active_learning_pi.al.evaluate_report import (
    collect_finetune_metrics,
    collect_iteration_context,
    write_cycle_report_md,
)
from active_learning_pi.al.inference_pool import predict_candidates
from active_learning_pi.al.paths import iteration_dir


PRE_FINETUNE_EVAL = "eval_off_anchor_pre_finetune.json"
POST_FINETUNE_EVAL = "eval_off_anchor_post_finetune.json"
POST_FINETUNE_SCORED = "scored_candidates_post_finetune.json"
ACQUISITION_RANK_QUALITY = "acquisition_rank_quality.json"
CYCLE_SUMMARY = "eval_cycle_summary.json"
LEGACY_EVAL = "eval_off_anchor.json"


def _off_anchor_mhz(cfg: dict) -> list[float]:
    return [float(x) for x in cfg.get("eval_off_anchor_mhz", [80, 250])]


def _eval_cfg(cfg: dict) -> dict:
    return cfg.get("evaluation") or {}


def evaluate_scored_vs_manifest(
    cfg: dict,
    manifest: list[dict[str, Any]],
    scored: list[dict[str, Any]],
) -> dict[str, Any]:
    report = evaluate_labels_vs_predictions(
        manifest,
        scored,
        off_anchor_mhz=_off_anchor_mhz(cfg),
    )
    ckpt = Path(cfg.get("checkpoint_path", ""))
    report["checkpoint_path"] = str(ckpt)
    if ckpt.is_file():
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        report["checkpoint_epoch"] = int(ck.get("epoch", 0))
        report["checkpoint_val_loss"] = float(ck.get("val_loss", float("nan")))
    return report


def compute_acquisition_rank_quality(
    cfg: dict,
    groot: Path,
    iteration: int,
    *,
    manifest: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Rank quality from acquisition-time ``scored_candidates.json`` vs ECAD labels."""
    it_dir = iteration_dir(cfg, iteration, groot)
    scored_path = it_dir / "scored_candidates.json"
    if not scored_path.is_file():
        return {"available": False, "reason": f"missing {scored_path}"}

    if manifest is None:
        manifest_path = it_dir / "ingest_manifest.json"
        if not manifest_path.is_file():
            return {"available": False, "reason": f"missing {manifest_path}"}
        manifest = load_json(manifest_path)

    scored = load_json(scored_path)
    selected = None
    sel_path = it_dir / "selected_for_simulation.json"
    if sel_path.is_file():
        selected = load_json(sel_path)

    report = evaluate_acquisition_rank_quality(
        manifest,
        scored,
        off_anchor_mhz=_off_anchor_mhz(cfg),
        selected=selected,
    )
    report["scored_path"] = str(scored_path)
    report["phase"] = "acquisition"
    save_eval_report(it_dir, ACQUISITION_RANK_QUALITY, report)
    return report


def save_eval_report(it_dir: Path, name: str, report: dict[str, Any]) -> Path:
    path = it_dir / name
    save_json(path, report)
    return path


def read_training_off_anchor_metrics(
    cfg: dict,
    groot: Path,
    *,
    epoch: int | None = None,
    kinds: tuple[str, ...] = ("layout_cross",),
) -> dict[str, Any]:
    """Latest rows from ``metrics/off_anchor_eval.csv`` for the experiment."""
    exp_rel = cfg.get("experiment_dir", "")
    csv_path = groot / exp_rel / "metrics" / "off_anchor_eval.csv"
    if not csv_path.is_file():
        return {"available": False, "path": str(csv_path), "rows": []}

    rows: list[dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if kinds and row.get("kind") not in kinds:
                continue
            rows.append({
                "epoch": int(float(row["epoch"])),
                "mhz": float(row["mhz"]),
                "kind": row["kind"],
                "n": int(float(row["n"])),
                "hm_fg_mse_mean": float(row["hm_fg_mse_mean"]),
                "pearson_fg_mean": float(row["pearson_fg_mean"]),
                "peak_loc_err_mean": float(row["peak_loc_err_mean"]),
            })

    if not rows:
        return {"available": False, "path": str(csv_path), "rows": []}

    target_epoch = epoch if epoch is not None else max(r["epoch"] for r in rows)
    epoch_rows = [r for r in rows if r["epoch"] == target_epoch]
    return {
        "available": True,
        "path": str(csv_path),
        "epoch": target_epoch,
        "rows": epoch_rows,
    }


def _ingested_candidates(cfg: dict, groot: Path, iteration: int) -> list[Any]:
    from active_learning_pi.al.candidates import Candidate

    it_dir = iteration_dir(cfg, iteration, groot)
    manifest = load_json(it_dir / "ingest_manifest.json")
    cand_raw = {int(c["candidate_id"]): c for c in load_json(it_dir / "candidates.json")}
    cands = []
    for m in manifest:
        if not m.get("ingest_ok"):
            continue
        cid = int(m["candidate_id"])
        raw = cand_raw.get(cid)
        if raw is None:
            continue
        cands.append(Candidate(**{k: raw[k] for k in ("candidate_id", "occupancy", "mhz", "k")}))
    return cands


def cmd_infer_post_finetune(cfg: dict, groot: Path, iteration: int) -> list[dict[str, Any]]:
    """Re-infer ingested ECAD layouts with the current checkpoint (post fine-tune)."""
    it_dir = iteration_dir(cfg, iteration, groot)
    eval_opts = _eval_cfg(cfg)
    scope = str(eval_opts.get("infer_scope", "ingested"))

    print(f"\n=== Post-finetune infer (iter {iteration}, scope={scope}) ===", flush=True)
    ckpt = groot / cfg["checkpoint_path"]
    print(f"  checkpoint: {ckpt}", flush=True)

    if scope == "all":
        from active_learning_pi.al.candidates import Candidate

        raw = load_json(it_dir / "candidates.json")
        cands = [
            Candidate(**{k: c[k] for k in ("candidate_id", "occupancy", "mhz", "k")})
            for c in raw
        ]
        print(f"  Re-scoring all {len(cands)} candidate(s)", flush=True)
        scored = predict_candidates(cfg, cands, groot)
        save_json(it_dir / POST_FINETUNE_SCORED, scored)
        return scored

    cands = _ingested_candidates(cfg, groot, iteration)
    if not cands:
        raise FileNotFoundError(
            f"No ingested labels for iter {iteration} — run ingest first ({it_dir})"
        )
    print(f"  Re-scoring {len(cands)} ingested layout(s)", flush=True)
    scored = predict_candidates(cfg, cands, groot)
    save_json(it_dir / POST_FINETUNE_SCORED, scored)
    print(f"  Saved → {it_dir / POST_FINETUNE_SCORED}", flush=True)
    return scored


def _delta(pre: float | None, post: float | None) -> float | None:
    if pre is None or post is None:
        return None
    return float(post) - float(pre)


def build_cycle_summary(
    cfg: dict,
    groot: Path,
    iteration: int,
    *,
    pre_report: dict[str, Any] | None,
    post_report: dict[str, Any],
    training_metrics: dict[str, Any] | None = None,
    acquisition_rank_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "iteration": iteration,
        "off_anchor_mhz": _off_anchor_mhz(cfg),
        "pre_finetune": pre_report,
        "post_finetune": post_report,
        "training_off_anchor": training_metrics,
        "acquisition_rank_quality": acquisition_rank_quality,
    }
    if pre_report and post_report:
        summary["improvement"] = {
            "p99_mae_delta": _delta(pre_report.get("p99_mae"), post_report.get("p99_mae")),
            "p99_median_ae_delta": _delta(
                pre_report.get("p99_median_ae"), post_report.get("p99_median_ae"),
            ),
            "n_pre": pre_report.get("n"),
            "n_post": post_report.get("n"),
        }
        pre_mae = pre_report.get("p99_mae")
        post_mae = post_report.get("p99_mae")
        if pre_mae and post_mae and pre_mae > 0:
            summary["improvement"]["p99_mae_pct"] = 100.0 * (float(pre_mae) - float(post_mae)) / float(pre_mae)
    return summary


def cmd_post_finetune_eval(cfg: dict, groot: Path, iteration: int) -> dict[str, Any]:
    """Re-infer ingested layouts + evaluate vs ECAD + write cycle summary."""
    it_dir = iteration_dir(cfg, iteration, groot)
    manifest_path = it_dir / "ingest_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing ingest manifest: {manifest_path}")

    eval_opts = _eval_cfg(cfg)
    manifest = load_json(manifest_path)

    pre_path = it_dir / PRE_FINETUNE_EVAL
    legacy_path = it_dir / LEGACY_EVAL
    pre_report = load_json(pre_path) if pre_path.is_file() else None
    if pre_report is None and legacy_path.is_file():
        pre_report = load_json(legacy_path)

    scored = cmd_infer_post_finetune(cfg, groot, iteration)
    post_report = evaluate_scored_vs_manifest(cfg, manifest, scored)
    post_report["phase"] = "post_finetune"
    save_eval_report(it_dir, POST_FINETUNE_EVAL, post_report)

    rank_q = compute_acquisition_rank_quality(cfg, groot, iteration, manifest=manifest)

    training_metrics = None
    if eval_opts.get("include_training_metrics", True):
        epoch = post_report.get("checkpoint_epoch")
        training_metrics = read_training_off_anchor_metrics(cfg, groot, epoch=epoch)

    end_ep = post_report.get("checkpoint_epoch")
    finetune_metrics = collect_finetune_metrics(cfg, groot, end_epoch=end_ep)
    cycle_context = collect_iteration_context(cfg, groot, iteration)

    summary = build_cycle_summary(
        cfg, groot, iteration,
        pre_report=pre_report,
        post_report=post_report,
        training_metrics=training_metrics,
        acquisition_rank_quality=rank_q,
    )
    summary["finetune"] = finetune_metrics
    summary["cycle_context"] = cycle_context
    save_json(it_dir / CYCLE_SUMMARY, summary)

    report_path = write_cycle_report_md(cfg, groot, iteration, summary)
    decision_path = write_decision_report(cfg, groot, iteration, summary=summary)

    print(f"\n=== Post-finetune evaluate (iter {iteration}) ===", flush=True)
    print(
        f"  n={post_report['n']}  p99_mae={post_report.get('p99_mae', 'n/a')}  "
        f"checkpoint_epoch={post_report.get('checkpoint_epoch', '?')}",
        flush=True,
    )
    if summary.get("improvement"):
        imp = summary["improvement"]
        delta = imp.get("p99_mae_delta")
        pct = imp.get("p99_mae_pct")
        if pct is not None:
            direction = "better" if pct >= 0 else "worse"
            print(
                f"  vs pre-finetune: p99_mae {delta:+.4f}  ({abs(pct):.1f}% {direction})",
                flush=True,
            )
        else:
            print(f"  vs pre-finetune: p99_mae delta {delta}", flush=True)
    if rank_q.get("available"):
        sp = rank_q.get("spearman_badness_vs_p99_err")
        print(
            f"  acquisition rank: n={rank_q.get('n')}  "
            f"spearman(badness,p99_err)={sp if sp is not None else 'n/a'}  "
            f"ranking_ok={rank_q.get('ranking_ok')}",
            flush=True,
        )
    print(f"  Summary → {it_dir / CYCLE_SUMMARY}", flush=True)
    print(f"  Report  → {report_path}", flush=True)
    print(f"  Decision→ {decision_path}", flush=True)
    return summary


def cmd_evaluate_pre_finetune(cfg: dict, groot: Path, iteration: int) -> dict[str, Any]:
    """Evaluate acquisition-time predictions vs ingested ECAD labels."""
    it_dir = iteration_dir(cfg, iteration, groot)
    manifest = load_json(it_dir / "ingest_manifest.json")
    scored = load_json(it_dir / "scored_candidates.json")
    report = evaluate_scored_vs_manifest(cfg, manifest, scored)
    report["phase"] = "pre_finetune"
    save_eval_report(it_dir, PRE_FINETUNE_EVAL, report)
    save_eval_report(it_dir, LEGACY_EVAL, report)
    rank_q = compute_acquisition_rank_quality(cfg, groot, iteration, manifest=manifest)

    # Partial cycle summary so decision ledger can update before FT finishes.
    partial = {
        "iteration": iteration,
        "off_anchor_mhz": _off_anchor_mhz(cfg),
        "pre_finetune": report,
        "post_finetune": None,
        "acquisition_rank_quality": rank_q,
        "cycle_context": collect_iteration_context(cfg, groot, iteration),
        "training_off_anchor": {"available": False},
    }
    save_json(it_dir / CYCLE_SUMMARY, partial)
    write_decision_report(cfg, groot, iteration, summary=partial)

    print(f"\n=== Pre-finetune evaluate (iter {iteration}) ===", flush=True)
    print(f"  n={report['n']}  p99_mae={report.get('p99_mae', 'n/a')}", flush=True)
    if rank_q.get("available"):
        sp = rank_q.get("spearman_badness_vs_p99_err")
        print(
            f"  acquisition rank: n={rank_q.get('n')}  "
            f"spearman(badness,p99_err)={sp if sp is not None else 'n/a'}  "
            f"ranking_ok={rank_q.get('ranking_ok')}",
            flush=True,
        )
    print(f"  Decision ledger updated (pre-FT; FT claims still UNCERTAIN)", flush=True)
    return report
