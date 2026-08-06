"""Markdown cycle evaluation report (pre/post fine-tune + training metrics)."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from active_learning_pi.al.config import load_json
from active_learning_pi.al.finetune_run import _load_yaml_like
from active_learning_pi.al.paths import iteration_dir, run_dir


CYCLE_REPORT_MD = "CYCLE_EVAL_REPORT.md"


def _fmt(v: Any, *, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if v != v:  # nan
            return "—"
        return f"{v:.{digits}f}"
    return str(v)


def _improvement_label(pre: float | None, post: float | None) -> str:
    if pre is None or post is None or pre == 0:
        return "—"
    pct = abs(100.0 * (float(pre) - float(post)) / float(pre))
    direction = "better" if float(post) < float(pre) else "worse"
    return f"{pct:.1f}% {direction}"


def read_loss_epoch_row(cfg: dict, groot: Path, epoch: int) -> dict[str, Any] | None:
    exp_rel = cfg.get("experiment_dir", "")
    path = groot / exp_rel / "metrics" / "loss.csv"
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(float(row["epoch"])) == epoch:
                return {
                    "epoch": epoch,
                    "train_total_loss": float(row["train_total_loss"]),
                    "train_heatmap_loss": float(row["train_heatmap_loss"]),
                    "val_total_loss": float(row["val_total_loss"]),
                    "val_heatmap_loss": float(row["val_heatmap_loss"]),
                }
    return None


def collect_finetune_metrics(cfg: dict, groot: Path, *, end_epoch: int | None) -> dict[str, Any]:
    ft = cfg.get("finetune") or {}
    extra = int(ft.get("extra_epochs", 50))
    end_ep = int(end_epoch) if end_epoch else None
    start_ep = (end_ep - extra) if end_ep else None

    out: dict[str, Any] = {
        "enabled": bool(ft.get("enabled")),
        "extra_epochs": extra,
        "start_epoch": start_ep,
        "end_epoch": end_ep,
        "checkpoint_path": str(ft.get("checkpoint_path", cfg.get("checkpoint_path", ""))),
        "overlay_data_dir": ft.get("overlay_data_dir") or cfg.get("overlay_data_dir"),
        "overlay_sample_weight": None,
    }

    runtime = groot / ft.get("experiment_dir", cfg.get("experiment_dir", "")) / "config_al_finetune.runtime.yaml"
    if runtime.is_file():
        try:
            data = _load_yaml_like(runtime)
            out["overlay_sample_weight"] = data.get("al_overlay_sample_weight")
            out["eval_off_anchor_mhz"] = data.get("eval_off_anchor_mhz")
            out["eval_off_anchor_mhz_weights"] = data.get("eval_off_anchor_mhz_weights")
            out["al_finetune_early_stop"] = data.get("al_finetune_early_stop")
            out["al_finetune_early_stop_patience"] = data.get("al_finetune_early_stop_patience")
        except Exception:
            pass

    exp_rel = ft.get("experiment_dir", cfg.get("experiment_dir", ""))
    best_oa = groot / exp_rel / "checkpoints" / "best_off_anchor_model.pt"
    out["best_off_anchor_checkpoint"] = str(best_oa) if best_oa.is_file() else None

    ckpt_path = groot / str(out["checkpoint_path"])
    if ckpt_path.is_file():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        out["end_val_loss"] = float(ck.get("val_loss", float("nan")))
        out["end_train_loss"] = float(ck.get("train_loss", float("nan")))

    if start_ep:
        start_row = read_loss_epoch_row(cfg, groot, start_ep)
        if start_row:
            out["start_val_loss"] = start_row["val_total_loss"]
            out["start_train_total_loss"] = start_row["train_total_loss"]
    if end_ep:
        end_row = read_loss_epoch_row(cfg, groot, end_ep)
        if end_row:
            out["end_val_loss_csv"] = end_row["val_total_loss"]
            out["end_train_total_loss"] = end_row["train_total_loss"]
            out["end_train_heatmap_loss"] = end_row["train_heatmap_loss"]
            out["end_val_heatmap_loss"] = end_row["val_heatmap_loss"]

    sv, ev = out.get("start_val_loss"), out.get("end_val_loss") or out.get("end_val_loss_csv")
    if sv is not None and ev is not None:
        out["val_loss_delta"] = float(ev) - float(sv)
    return out


def collect_iteration_context(cfg: dict, groot: Path, iteration: int) -> dict[str, Any]:
    it_dir = iteration_dir(cfg, iteration, groot)
    ctx: dict[str, Any] = {"iteration": iteration, "iteration_dir": str(it_dir)}

    cand_path = it_dir / "candidates.json"
    if cand_path.is_file():
        ctx["num_candidates"] = len(load_json(cand_path))

    sel_path = it_dir / "selected_for_simulation.json"
    if sel_path.is_file():
        selected = load_json(sel_path)
        ctx["num_selected"] = len(selected)
        if selected:
            ctx["badness_min"] = float(selected[-1].get("badness", 0))
            ctx["badness_max"] = float(selected[0].get("badness", 0))

    manifest_path = it_dir / "ingest_manifest.json"
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        ctx["num_ingest_total"] = len(manifest)
        ctx["num_ingest_ok"] = sum(1 for m in manifest if m.get("ingest_ok"))
        from collections import Counter
        ctx["ingested_k_distribution"] = dict(
            sorted(Counter(int(m.get("k", 0)) for m in manifest if m.get("ingest_ok")).items())
        )

    overlay_report = run_dir(cfg, groot) / "overlay_build_report.json"
    if overlay_report.is_file():
        ob = load_json(overlay_report)
        ctx["overlay_total"] = int(ob.get("candidates_seen", 0))
        ctx["overlay_added_last"] = int(ob.get("added", 0))
        ctx["overlay_root"] = ob.get("overlay_root")

    return ctx


def _row_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _per_sample_rows(
    pre_rows: list[dict[str, Any]] | None,
    post_rows: list[dict[str, Any]] | None,
) -> list[list[str]]:
    pre_map = {int(r["candidate_id"]): r for r in (pre_rows or [])}
    post_map = {int(r["candidate_id"]): r for r in (post_rows or [])}
    ids = sorted(set(pre_map) | set(post_map))
    out = []
    for cid in ids:
        pre = pre_map.get(cid, {})
        post = post_map.get(cid, {})
        mhz = pre.get("mhz") or post.get("mhz")
        out.append([
            str(cid),
            _fmt(mhz, digits=1),
            _fmt(pre.get("real_p99") or post.get("real_p99")),
            _fmt(pre.get("pred_p99_mean")),
            _fmt(pre.get("p99_abs_err")),
            _fmt(post.get("pred_p99_mean")),
            _fmt(post.get("p99_abs_err")),
        ])
    return out


def build_cycle_report_markdown(
    cfg: dict,
    groot: Path,
    summary: dict[str, Any],
    *,
    finetune: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> str:
    it = int(summary["iteration"])
    run_name = cfg.get("run_name", "al_run")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pre = summary.get("pre_finetune") or {}
    post = summary.get("post_finetune") or {}
    imp = summary.get("improvement") or {}
    train_oa = summary.get("training_off_anchor") or {}
    finetune = finetune or summary.get("finetune") or {}
    context = context or summary.get("cycle_context") or {}
    mhz_list = ", ".join(str(int(m)) if float(m).is_integer() else str(m) for m in summary.get("off_anchor_mhz", []))

    lines = [
        f"# AL Cycle Evaluation Report — iteration {it}",
        "",
        f"**Run:** `{run_name}`  ",
        f"**Generated:** {now}  ",
        f"**Experiment:** `{cfg.get('experiment_dir', '')}`  ",
        f"**Acquisition mode:** `{cfg.get('acquisition_mode', 'mc')}`  ",
        f"**Off-anchor MHz evaluated:** {mhz_list}",
        "",
        "---",
        "",
        "## 0. Decision checklist (see also `DECISION_REPORT.md`)",
        "",
        "_Numerical claims are judged after collecting cycle evals + equal-budget A/B. "
        "Regenerate with `COMMAND=evaluate-decision`._",
        "",
    ]

    # Inline compact claim preview if ledger already exists.
    try:
        from active_learning_pi.al.decision_report import (
            DECISION_LEDGER,
            build_decision_ledger,
        )

        ledger = build_decision_ledger(cfg, groot, it, summary=summary)
        rollup = ledger.get("rollup") or {}
        lines.append(f"**Overall:** **{rollup.get('overall', 'UNCERTAIN')}** — {rollup.get('summary', '')}")
        lines.append("")
        claim_rows = [
            [c["status"], c["id"], (c.get("note") or c["statement"])[:120]]
            for c in (ledger.get("claims") or [])
        ]
        lines.append(_row_table(["Status", "Claim", "Note"], claim_rows))
        lines.append("")
        lines.append(f"_Machine-readable:_ `{run_dir(cfg, groot) / DECISION_LEDGER}`")
    except Exception as exc:
        lines.append(f"_Decision ledger unavailable ({exc})._")

    lines.extend([
        "",
        "---",
        "",
        "## 1. Cycle overview",
        "",
    ])

    overview_rows = [
        ["Candidates generated", str(context.get("num_candidates", "—"))],
        ["Selected for ECAD", str(context.get("num_selected", "—"))],
        ["Ingested heatmaps (ok/total)", f"{context.get('num_ingest_ok', '—')} / {context.get('num_ingest_total', '—')}"],
        ["Overlay samples (cumulative)", str(context.get("overlay_total", "—"))],
        ["Overlay added (last build)", str(context.get("overlay_added_last", "—"))],
    ]
    if context.get("badness_min") is not None:
        overview_rows.append(["Selection badness range", f"{_fmt(context['badness_min'], digits=6)} … {_fmt(context['badness_max'], digits=6)}"])
    k_vals = cfg.get("k_values") or (
        list(range(int(cfg["k_min"]), int(cfg["k_max"]) + 1))
        if cfg.get("k_min") is not None and cfg.get("k_max") is not None
        else [int(cfg.get("fixed_k", 5))]
    )
    overview_rows.append(["K range", ", ".join(str(k) for k in k_vals)])
    if context.get("ingested_k_distribution"):
        overview_rows.append(["Ingested K distribution", str(context["ingested_k_distribution"])])
    from active_learning_pi.al.per_k_acquire import load_k_sweep_summary
    ksum = load_k_sweep_summary(cfg, int(summary["iteration"]), groot)
    if ksum:
        overview_rows.append(["Per-K pools", f"{ksum['candidates_per_k']} candidates → worst {ksum['worst_per_k']} each"])
        overview_rows.append(["Total ECAD selected", str(ksum.get("total_selected", "—"))])
        if ksum.get("mhz_strata_per_k"):
            overview_rows.append(["MHz strata / K (pool)", str(ksum["mhz_strata_per_k"])])
        if ksum.get("selected_mhz_distribution"):
            overview_rows.append(["ECAD MHz distribution", str(ksum["selected_mhz_distribution"])])
    lines.append(_row_table(["Metric", "Value"], overview_rows))
    lines.extend(["", "---", "", "## 2. Fine-tune"])

    if finetune.get("enabled"):
        ft_rows = [
            ["Start epoch", str(finetune.get("start_epoch", "—"))],
            ["End epoch", str(finetune.get("end_epoch", "—"))],
            ["Extra epochs", str(finetune.get("extra_epochs", "—"))],
            ["Checkpoint", f"`{finetune.get('checkpoint_path', '')}`"],
            ["Overlay dataset", f"`{finetune.get('overlay_data_dir', '')}`"],
            ["Overlay sample weight", _fmt(finetune.get("overlay_sample_weight"), digits=1)],
            ["Off-anchor eval MHz", ", ".join(str(x) for x in (finetune.get("eval_off_anchor_mhz") or [])) or "—"],
            ["Off-anchor MHz weights", str(finetune.get("eval_off_anchor_mhz_weights") or "—")],
            ["Early stop (off-anchor MSE)", str(finetune.get("al_finetune_early_stop", "—"))],
            ["Early-stop patience", str(finetune.get("al_finetune_early_stop_patience", "—"))],
            ["Best off-anchor checkpoint", f"`{finetune.get('best_off_anchor_checkpoint', '')}`" if finetune.get("best_off_anchor_checkpoint") else "—"],
            ["Val loss (start)", _fmt(finetune.get("start_val_loss"))],
            ["Val loss (end)", _fmt(finetune.get("end_val_loss") or finetune.get("end_val_loss_csv"))],
            ["Val loss Δ", _fmt(finetune.get("val_loss_delta"))],
            ["Train total loss (final ep)", _fmt(finetune.get("end_train_total_loss"))],
            ["Train heatmap loss (final ep)", _fmt(finetune.get("end_train_heatmap_loss"))],
            ["Val heatmap loss (final ep)", _fmt(finetune.get("end_val_heatmap_loss"))],
        ]
        lines.append(_row_table(["Metric", "Value"], ft_rows))
    else:
        lines.append("_Fine-tune disabled for this run._")

    rank_q = summary.get("acquisition_rank_quality") or {}
    lines.extend(["", "---", "", "## 3. Acquisition ranking quality", ""])
    lines.append(
        "_Does acquisition-time **badness** rank true ECAD **p99 abs error**? "
        "(positive Spearman ⇒ scoring is directionally correct.)_"
    )
    lines.append("")
    if rank_q.get("available"):
        rq_rows = [
            ["n (labeled off-anchor)", str(rank_q.get("n", "—"))],
            ["Spearman(badness, p99_err)", _fmt(rank_q.get("spearman_badness_vs_p99_err"))],
            ["Spearman stratified (per-MHz)", _fmt(rank_q.get("spearman_stratified_mhz"))],
            ["Spearman (MHz-zscored badness)", _fmt(rank_q.get("spearman_mhz_zscored_badness"))],
            ["Pearson(badness, p99_err)", _fmt(rank_q.get("pearson_badness_vs_p99_err"))],
            ["Ranking OK", str(rank_q.get("ranking_ok", "—"))],
            ["Mean p99 err (all labeled)", _fmt(rank_q.get("mean_p99_err_all_labeled"))],
            ["Mean p99 err (top half badness)", _fmt(rank_q.get("mean_p99_err_top_half_badness"))],
            ["Mean p99 err (bottom half badness)", _fmt(rank_q.get("mean_p99_err_bottom_half_badness"))],
            ["Top / bottom half lift", _fmt(rank_q.get("top_vs_bottom_half_lift"))],
            ["n selected ∩ labeled", str(rank_q.get("n_selected_labeled", "—"))],
            ["Mean p99 err (selected)", _fmt(rank_q.get("mean_p99_err_selected"))],
            ["Selected / all lift", _fmt(rank_q.get("selected_vs_all_lift"))],
            ["Mean p99 err (top-k by badness)", _fmt(rank_q.get("mean_p99_err_top_k_by_badness"))],
        ]
        lines.append(_row_table(["Metric", "Value"], rq_rows))
        per_mhz = rank_q.get("per_mhz") or []
        if per_mhz:
            lines.extend(["", "### Per-MHz acquisition Spearman", ""])
            mhz_rows = [
                [
                    _fmt(r.get("mhz"), digits=1),
                    str(r.get("n", "—")),
                    _fmt(r.get("spearman")),
                ]
                for r in per_mhz
            ]
            lines.append(_row_table(["MHz", "n", "Spearman"], mhz_rows))
    else:
        reason = rank_q.get("reason") or "not available"
        lines.append(f"_Acquisition rank quality not available ({reason})._")

    lines.extend(["", "---", "", "## 4. ECAD label evaluation (physical p99)", ""])
    ecad_summary = [
        ["Pre-finetune", str(pre.get("n", "—")), _fmt(pre.get("p99_mae")), _fmt(pre.get("p99_median_ae"))],
        ["Post-finetune", str(post.get("n", "—")), _fmt(post.get("p99_mae")), _fmt(post.get("p99_median_ae"))],
    ]
    lines.append(_row_table(["Phase", "n", "p99 MAE", "p99 median AE"], ecad_summary))

    if imp:
        lines.extend([
            "",
            "### Pre → post change",
            "",
            f"- **p99 MAE Δ:** {_fmt(imp.get('p99_mae_delta'))} ({_improvement_label(pre.get('p99_mae'), post.get('p99_mae'))} vs pre-finetune)",
            f"- **p99 median AE Δ:** {_fmt(imp.get('p99_median_ae_delta'))}",
            "",
        ])

    sample_rows = _per_sample_rows(pre.get("rows"), post.get("rows"))
    if sample_rows:
        lines.extend([
            "### Per-layout detail (off-anchor MHz only)",
            "",
            _row_table(
                ["candidate_id", "MHz", "real p99", "pre pred", "pre abs err", "post pred", "post abs err"],
                sample_rows,
            ),
            "",
        ])

    lines.extend(["---", "", "## 5. Training validation — off-anchor (`layout_cross`)", ""])
    if train_oa.get("available") and train_oa.get("rows"):
        lines.append(f"_Epoch {train_oa.get('epoch')} — from `{train_oa.get('path')}`_")
        lines.append("")
        oa_rows = [
            [
                _fmt(r["mhz"], digits=1),
                _fmt(r["hm_fg_mse_mean"]),
                _fmt(r["pearson_fg_mean"]),
                _fmt(r["peak_loc_err_mean"]),
                str(r["n"]),
            ]
            for r in sorted(train_oa["rows"], key=lambda x: x["mhz"])
        ]
        lines.append(_row_table(["MHz", "FG MSE", "Pearson", "peak loc err", "n"], oa_rows))
    else:
        lines.append("_Training off-anchor metrics not available._")

    it_dir = iteration_dir(cfg, it, groot)
    lines.extend([
        "",
        "---",
        "",
        "## 6. Artifacts",
        "",
        f"- `{it_dir / 'acquisition_rank_quality.json'}`",
        f"- `{it_dir / 'eval_off_anchor_pre_finetune.json'}`",
        f"- `{it_dir / 'eval_off_anchor_post_finetune.json'}`",
        f"- `{it_dir / 'eval_cycle_summary.json'}`",
        f"- `{it_dir / 'scored_candidates_post_finetune.json'}`",
        f"- `{it_dir / CYCLE_REPORT_MD}`",
        f"- `{run_dir(cfg, groot) / 'DECISION_REPORT.md'}`",
        f"- `{run_dir(cfg, groot) / 'decision_ledger.json'}`",
        "",
    ])
    return "\n".join(lines)


def write_cycle_report_md(
    cfg: dict,
    groot: Path,
    iteration: int,
    summary: dict[str, Any],
) -> Path:
    """Write markdown report under iteration dir and copy pointer at run root."""
    eval_opts = (cfg.get("evaluation") or {})
    if not eval_opts.get("write_markdown_report", True):
        return iteration_dir(cfg, iteration, groot) / CYCLE_REPORT_MD

    if "finetune" not in summary:
        summary = dict(summary)
        end_ep = (summary.get("post_finetune") or {}).get("checkpoint_epoch")
        summary["finetune"] = collect_finetune_metrics(cfg, groot, end_epoch=end_ep)
    if "cycle_context" not in summary:
        summary = dict(summary)
        summary["cycle_context"] = collect_iteration_context(cfg, groot, iteration)

    md = build_cycle_report_markdown(cfg, groot, summary)
    it_dir = iteration_dir(cfg, iteration, groot)
    out = it_dir / CYCLE_REPORT_MD
    out.write_text(md, encoding="utf-8")

    latest = run_dir(cfg, groot) / f"LATEST_CYCLE_EVAL_REPORT_iter{iteration:04d}.md"
    latest.write_text(md, encoding="utf-8")
    return out


def cmd_write_cycle_report(cfg: dict, groot: Path, iteration: int) -> Path:
    """Regenerate markdown report from existing ``eval_cycle_summary.json``."""
    it_dir = iteration_dir(cfg, iteration, groot)
    summary_path = it_dir / "eval_cycle_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing {summary_path} — run evaluate-post-finetune first")
    summary = load_json(summary_path)
    # Backfill rank quality for summaries written before this metric existed.
    rq = summary.get("acquisition_rank_quality")
    if not rq or not rq.get("available"):
        from active_learning_pi.al.evaluate_cycle import compute_acquisition_rank_quality

        rank_q = compute_acquisition_rank_quality(cfg, groot, iteration)
        summary = dict(summary)
        summary["acquisition_rank_quality"] = rank_q
        from active_learning_pi.al.config import save_json

        save_json(summary_path, summary)
    path = write_cycle_report_md(cfg, groot, iteration, summary)
    print(f"\n=== Cycle evaluation report (iter {iteration}) ===", flush=True)
    print(f"  → {path}", flush=True)
    return path
