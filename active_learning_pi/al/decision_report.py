"""Decision ledger: collect numerical evals → PASS/FAIL/UNCERTAIN claims.

Every AL workflow step that produces numbers should feed this ledger. At the end
of a cycle (or after acquisition A/B), write a decision report that answers:

- Did acquisition / uncertainty scoring point in the right direction?
- Did fine-tune improve physical (ECAD) metrics?
- Is the equal-budget A/B claim (GP vs random / MC) supported?
- Are sample sizes / budgets sufficient to trust the claim?
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from active_learning_pi.al.config import load_json, save_json
from active_learning_pi.al.paths import iteration_dir, run_dir

DECISION_LEDGER = "decision_ledger.json"
DECISION_REPORT_MD = "DECISION_REPORT.md"
CYCLE_SUMMARY = "eval_cycle_summary.json"
ACQUISITION_RANK = "acquisition_rank_quality.json"
DEFAULT_AB_LATEST = (
    "active_learning_pi/runs/acquisition_ab_validation_exp059/LATEST_acquisition_ab.json"
)
DEFAULT_HOLE_FINDING = (
    "active_learning_pi/runs/hole_finding_exp059/LATEST_hole_finding.json"
)


def _status(ok: bool | None, *, missing: bool = False) -> str:
    if missing or ok is None:
        return "UNCERTAIN"
    return "PASS" if ok else "FAIL"


def _fmt(v: Any, *, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if v != v:
            return "—"
        return f"{v:.{digits}f}"
    return str(v)


def _load_optional(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return load_json(path)
    except Exception:
        return None


def _claim(
    claim_id: str,
    statement: str,
    status: str,
    *,
    evidence: dict[str, Any] | None = None,
    note: str = "",
) -> dict[str, Any]:
    return {
        "id": claim_id,
        "statement": statement,
        "status": status,  # PASS | FAIL | UNCERTAIN
        "evidence": evidence or {},
        "note": note,
    }


def build_claims_from_cycle(
    summary: dict[str, Any],
    *,
    acquisition_ab: dict[str, Any] | None = None,
    hole_finding: dict[str, Any] | None = None,
    min_rank_n: int = 30,
    min_ecad_n: int = 20,
) -> list[dict[str, Any]]:
    """Turn one cycle's numericals into explicit decision claims."""
    claims: list[dict[str, Any]] = []
    rank = summary.get("acquisition_rank_quality") or {}
    pre = summary.get("pre_finetune") or {}
    post = summary.get("post_finetune") or {}
    imp = summary.get("improvement") or {}
    ctx = summary.get("cycle_context") or {}
    train_oa = summary.get("training_off_anchor") or {}

    # --- Claim: acquisition scoring direction ---
    n_rank = int(rank.get("n") or 0) if rank.get("available") else 0
    sp = rank.get("spearman_badness_vs_p99_err")
    sp_strat = rank.get("spearman_stratified_mhz")
    sp_mhz_z = rank.get("spearman_mhz_zscored_badness")
    lift = rank.get("top_vs_bottom_half_lift")
    ranking_ok = rank.get("ranking_ok")
    if not rank.get("available"):
        claims.append(
            _claim(
                "acq_direction",
                "Acquisition / uncertainty scores rank true ECAD error (Spearman > 0).",
                "UNCERTAIN",
                note=rank.get("reason") or "acquisition_rank_quality missing",
            )
        )
    elif n_rank < min_rank_n:
        claims.append(
            _claim(
                "acq_direction",
                "Acquisition / uncertainty scores rank true ECAD error (Spearman > 0).",
                "UNCERTAIN",
                evidence={
                    "n": n_rank,
                    "spearman": sp,
                    "spearman_stratified_mhz": sp_strat,
                    "spearman_mhz_zscored_badness": sp_mhz_z,
                    "top_bottom_lift": lift,
                },
                note=f"n={n_rank} < min_rank_n={min_rank_n}; direction signal underpowered",
            )
        )
    else:
        ok = bool(ranking_ok)
        claims.append(
            _claim(
                "acq_direction",
                "Acquisition / uncertainty scores rank true ECAD error (Spearman > 0).",
                _status(ok),
                evidence={
                    "n": n_rank,
                    "spearman_badness_vs_p99_err": sp,
                    "spearman_stratified_mhz": sp_strat,
                    "spearman_mhz_zscored_badness": sp_mhz_z,
                    "pearson_badness_vs_p99_err": rank.get("pearson_badness_vs_p99_err"),
                    "top_vs_bottom_half_lift": lift,
                    "selected_vs_all_lift": rank.get("selected_vs_all_lift"),
                    "ranking_ok": ranking_ok,
                    "per_mhz": rank.get("per_mhz"),
                },
                note=(
                    "Scoring is directionally correct on labeled off-anchor "
                    "(overall and/or MHz-stratified)."
                    if ok
                    else "Scoring does NOT beat chance on labeled off-anchor — do not trust AL selection."
                ),
            )
        )

    # --- Claim: equal-budget GP vs random (optional A/B artifact) ---
    if acquisition_ab is None:
        claims.append(
            _claim(
                "acq_ab_gp_vs_random",
                "Equal-budget GP(mu) beats random on holdout (top-k true residual + Spearman>0).",
                "UNCERTAIN",
                note="No acquisition A/B JSON linked — run validate_acquisition_ab.py",
            )
        )
    else:
        passed = bool(acquisition_ab.get("pass_gp_beats_random"))
        methods = acquisition_ab.get("methods") or {}
        gp = methods.get("gp_mu_only") or {}
        rnd = methods.get("random") or {}
        claims.append(
            _claim(
                "acq_ab_gp_vs_random",
                "Equal-budget GP(mu) beats random on holdout (top-k true residual + Spearman>0).",
                _status(passed),
                evidence={
                    "verdict": acquisition_ab.get("verdict"),
                    "gp_top_k_mean_true_y": gp.get("top_k_mean_true_y"),
                    "gp_spearman": gp.get("spearman_vs_true_y"),
                    "gp_lift_vs_random": gp.get("lift_vs_random_top_k"),
                    "random_top_k_mean_true_y": rnd.get("top_k_mean_true_y"),
                    "budget_k": acquisition_ab.get("budget_k"),
                    "n_test": acquisition_ab.get("n_test"),
                    "experiment": acquisition_ab.get("experiment"),
                },
                note=str(acquisition_ab.get("verdict") or ""),
            )
        )
        mc = methods.get("mc_auto_bad")
        if mc is None:
            claims.append(
                _claim(
                    "acq_ab_gp_vs_mc",
                    "Equal-budget GP beats MC auto_bad on the same holdout subset.",
                    "UNCERTAIN",
                    note="MC not included (INCLUDE_MC=False) — optional when GPU free",
                )
            )
        else:
            gp_sub = methods.get("gp_ucb_novelty_on_mc_subset") or methods.get("gp_mu_on_mc_subset") or {}
            ratio = gp_sub.get("lift_vs_mc_top_k")
            ok_mc = ratio is not None and float(ratio) >= 1.0
            claims.append(
                _claim(
                    "acq_ab_gp_vs_mc",
                    "Equal-budget GP beats MC auto_bad on the same holdout subset.",
                    _status(ok_mc),
                    evidence={
                        "mc_top_k": mc.get("top_k_mean_true_y"),
                        "mc_spearman": mc.get("spearman_vs_true_y"),
                        "gp_vs_mc_ratio": ratio,
                        "n_mc": mc.get("n_mc"),
                    },
                )
            )

    # --- Claim: solid hole-finding enrichment (labeled holdout) ---
    if hole_finding is None:
        claims.append(
            _claim(
                "hole_finding",
                "GP acquisition enriches true high-error holes vs random (labeled holdout).",
                "UNCERTAIN",
                note="No hole-finding JSON — run evaluate_hole_finding.py",
            )
        )
    else:
        verd = (hole_finding.get("verdict") or {})
        overall = str(verd.get("overall", "UNCERTAIN"))
        primary = verd.get("primary_budget")
        budgets = hole_finding.get("budgets") or {}
        prim_m = budgets.get(primary or "", {})
        status = (
            "PASS" if overall == "PASS"
            else ("FAIL" if overall == "FAIL" else "UNCERTAIN")
        )
        claims.append(
            _claim(
                "hole_finding",
                "GP acquisition enriches true high-error holes vs random (labeled holdout).",
                status,
                evidence={
                    "overall": overall,
                    "primary_budget": primary,
                    "lift_vs_random": prim_m.get("lift_vs_random"),
                    "pool_enrichment": prim_m.get("pool_enrichment"),
                    "hole_capture_rate": prim_m.get("hole_capture_rate"),
                    "hole_capture_lift_vs_chance": prim_m.get("hole_capture_lift_vs_chance"),
                    "spearman": prim_m.get("spearman_score_vs_y"),
                    "checks": verd.get("checks"),
                    "midband": (hole_finding.get("midband_structured") or {}).get("lift_vs_random"),
                },
                note=str(verd.get("summary") or ""),
            )
        )

    # --- Claim: fine-tune improved physical ECAD accuracy ---
    n_pre = int(pre.get("n") or 0)
    n_post = int(post.get("n") or 0)
    mae_delta = imp.get("p99_mae_delta")
    mae_pct = imp.get("p99_mae_pct")
    if not pre or not post or mae_delta is None:
        claims.append(
            _claim(
                "ft_physical_improve",
                "Fine-tune reduced off-anchor physical p99 MAE vs pre-finetune.",
                "UNCERTAIN",
                note="Need both pre- and post-finetune ECAD evals",
            )
        )
    elif min(n_pre, n_post) < min_ecad_n:
        claims.append(
            _claim(
                "ft_physical_improve",
                "Fine-tune reduced off-anchor physical p99 MAE vs pre-finetune.",
                "UNCERTAIN",
                evidence={"n_pre": n_pre, "n_post": n_post, "p99_mae_delta": mae_delta},
                note=f"n < min_ecad_n={min_ecad_n}",
            )
        )
    else:
        # lower MAE is better → mae_pct > 0 means (pre-post)/pre > 0
        improved = (mae_pct is not None and float(mae_pct) > 0) or (
            mae_pct is None and float(mae_delta) < 0
        )
        claims.append(
            _claim(
                "ft_physical_improve",
                "Fine-tune reduced off-anchor physical p99 MAE vs pre-finetune.",
                _status(improved),
                evidence={
                    "n_pre": n_pre,
                    "n_post": n_post,
                    "p99_mae_pre": pre.get("p99_mae"),
                    "p99_mae_post": post.get("p99_mae"),
                    "p99_mae_delta": mae_delta,
                    "p99_mae_pct": mae_pct,
                    "p99_median_ae_delta": imp.get("p99_median_ae_delta"),
                },
                note=(
                    "Physical accuracy improved on labeled AL layouts."
                    if improved
                    else "Physical p99 MAE did not improve — FT claim not supported on this cycle."
                ),
            )
        )

    # --- Claim: evaluation trust / budget hygiene ---
    n_sel = ctx.get("num_selected")
    n_ok = ctx.get("num_ingest_ok")
    budget_ok = None
    if n_sel is not None and n_ok is not None and int(n_sel) > 0:
        budget_ok = int(n_ok) >= int(0.9 * int(n_sel))
    claims.append(
        _claim(
            "eval_budget_trust",
            "ECAD ingest covered ≥90% of selected budget (evaluation not starved).",
            _status(budget_ok, missing=budget_ok is None),
            evidence={
                "num_selected": n_sel,
                "num_ingest_ok": n_ok,
                "num_candidates": ctx.get("num_candidates"),
                "off_anchor_mhz": summary.get("off_anchor_mhz"),
            },
            note=(
                "Ingest coverage OK."
                if budget_ok
                else "Missing selection/ingest counts or low ingest rate — treat metrics cautiously."
            ),
        )
    )

    # --- Claim: training off-anchor metrics present (hygiene) ---
    claims.append(
        _claim(
            "train_off_anchor_logged",
            "Training off-anchor (layout_cross) metrics were logged for this checkpoint epoch.",
            _status(bool(train_oa.get("available"))),
            evidence={
                "epoch": train_oa.get("epoch"),
                "n_mhz_rows": len(train_oa.get("rows") or []),
                "path": train_oa.get("path"),
            },
            note=(
                "Use FG MSE / Pearson trends across cycles as secondary evidence."
                if train_oa.get("available")
                else "Missing metrics/off_anchor_eval.csv rows — training claim incomplete."
            ),
        )
    )

    return claims


def aggregate_verdict(claims: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll-up: blocking FAILs vs soft UNCERTAIN gaps."""
    by = {c["id"]: c["status"] for c in claims}
    blocking = ["acq_direction", "ft_physical_improve", "acq_ab_gp_vs_random", "hole_finding"]
    fails = [cid for cid in blocking if by.get(cid) == "FAIL"]
    uncertain = [c["id"] for c in claims if c["status"] == "UNCERTAIN"]
    passes = [c["id"] for c in claims if c["status"] == "PASS"]

    if fails:
        overall = "FAIL"
        summary = (
            "Do not claim improvement / correct uncertainty until FAIL claims are resolved: "
            + ", ".join(fails)
        )
    elif (
        "acq_direction" in uncertain
        or "ft_physical_improve" in uncertain
        or "hole_finding" in uncertain
    ):
        overall = "UNCERTAIN"
        summary = (
            "Incomplete evidence — finish ECAD eval + acquisition A/B + hole-finding "
            f"before thesis claims. UNCERTAIN: {', '.join(uncertain)}"
        )
    else:
        overall = "PASS"
        summary = (
            "Numerical evidence supports directional acquisition + FT improvement "
            f"(PASS: {', '.join(passes)})."
        )
        if uncertain:
            summary += f" Remaining optional gaps: {', '.join(uncertain)}."

    return {
        "overall": overall,
        "summary": summary,
        "pass_ids": passes,
        "fail_ids": fails,
        "uncertain_ids": uncertain,
    }


def resolve_acquisition_ab(
    groot: Path,
    cfg: dict,
    *,
    explicit_path: str | Path | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Find latest equal-budget A/B JSON (config path → default exp059 path)."""
    eval_opts = cfg.get("evaluation") or {}
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    ab_cfg = eval_opts.get("acquisition_ab_path")
    if ab_cfg:
        candidates.append(groot / str(ab_cfg) if not Path(ab_cfg).is_absolute() else Path(ab_cfg))
    candidates.append(run_dir(cfg, groot) / "LATEST_acquisition_ab.json")
    candidates.append(groot / DEFAULT_AB_LATEST)

    for p in candidates:
        data = _load_optional(p)
        if data is not None:
            return data, str(p)
    return None, None


def resolve_hole_finding(
    groot: Path,
    cfg: dict,
) -> tuple[dict[str, Any] | None, str | None]:
    eval_opts = cfg.get("evaluation") or {}
    candidates: list[Path] = []
    hf_cfg = eval_opts.get("hole_finding_path")
    if hf_cfg:
        candidates.append(groot / str(hf_cfg) if not Path(hf_cfg).is_absolute() else Path(hf_cfg))
    candidates.append(run_dir(cfg, groot) / "LATEST_hole_finding.json")
    candidates.append(groot / DEFAULT_HOLE_FINDING)
    for p in candidates:
        data = _load_optional(p)
        if data is not None:
            return data, str(p)
    return None, None


def build_decision_ledger(
    cfg: dict,
    groot: Path,
    iteration: int,
    *,
    summary: dict[str, Any] | None = None,
    acquisition_ab: dict[str, Any] | None = None,
    ab_path: str | None = None,
    hole_finding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    it_dir = iteration_dir(cfg, iteration, groot)
    if summary is None:
        summary = _load_optional(it_dir / CYCLE_SUMMARY) or {}
        if not summary.get("acquisition_rank_quality"):
            rq = _load_optional(it_dir / ACQUISITION_RANK)
            if rq:
                summary = dict(summary)
                summary["acquisition_rank_quality"] = rq

    resolved_ab_path = ab_path
    if acquisition_ab is None:
        acquisition_ab, resolved_ab_path = resolve_acquisition_ab(groot, cfg)

    hole_path = None
    if hole_finding is None:
        hole_finding, hole_path = resolve_hole_finding(groot, cfg)

    eval_opts = cfg.get("evaluation") or {}
    claims = build_claims_from_cycle(
        summary,
        acquisition_ab=acquisition_ab,
        hole_finding=hole_finding,
        min_rank_n=int(eval_opts.get("decision_min_rank_n", 30)),
        min_ecad_n=int(eval_opts.get("decision_min_ecad_n", 20)),
    )
    rollup = aggregate_verdict(claims)

    return {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "run_name": cfg.get("run_name"),
        "experiment_dir": cfg.get("experiment_dir"),
        "iteration": iteration,
        "acquisition_mode": cfg.get("acquisition_mode", "mc"),
        "acquisition_ab_path": resolved_ab_path,
        "hole_finding_path": hole_path,
        "claims": claims,
        "rollup": rollup,
        "key_numbers": {
            "acquisition_rank": summary.get("acquisition_rank_quality"),
            "improvement": summary.get("improvement"),
            "pre_p99_mae": (summary.get("pre_finetune") or {}).get("p99_mae"),
            "post_p99_mae": (summary.get("post_finetune") or {}).get("p99_mae"),
            "hole_finding_verdict": (hole_finding or {}).get("verdict"),
            "cycle_context": {
                k: (summary.get("cycle_context") or {}).get(k)
                for k in (
                    "num_candidates",
                    "num_selected",
                    "num_ingest_ok",
                    "num_ingest_total",
                    "overlay_total",
                )
            },
        },
    }


def render_decision_markdown(ledger: dict[str, Any]) -> str:
    rollup = ledger.get("rollup") or {}
    claims = ledger.get("claims") or []
    kn = ledger.get("key_numbers") or {}
    lines = [
        f"# Decision report — `{ledger.get('run_name')}` iter {ledger.get('iteration')}",
        "",
        f"**Generated:** {ledger.get('generated_utc')}  ",
        f"**Experiment:** `{ledger.get('experiment_dir')}`  ",
        f"**Acquisition mode:** `{ledger.get('acquisition_mode')}`  ",
        f"**Overall:** **{rollup.get('overall', 'UNCERTAIN')}**  ",
        "",
        rollup.get("summary", ""),
        "",
        "---",
        "",
        "## How to use this report",
        "",
        "Every workflow stage must leave **numerical** artifacts. This ledger turns them into",
        "claims you can accept or reject:",
        "",
        "1. **acq_direction** — is uncertainty / GP scoring pointing at true error?",
        "2. **acq_ab_gp_vs_random** — equal-budget GP vs random (and MC if present)?",
        "3. **hole_finding** — does GP top-K enrich true high-error holes vs random?",
        "4. **ft_physical_improve** — did fine-tune actually improve physical p99?",
        "5. **eval_budget_trust** — was the ECAD budget fully evaluated?",
        "6. **train_off_anchor_logged** — did training log off-anchor numbers?",
        "",
        "Only claim a thesis result when blocking claims are **PASS** (or explicitly scoped).",
        "",
        "---",
        "",
        "## Claim checklist",
        "",
        "| Status | Claim | Key evidence |",
        "| --- | --- | --- |",
    ]
    for c in claims:
        ev = c.get("evidence") or {}
        bits = []
        for k in (
            "spearman_badness_vs_p99_err",
            "top_vs_bottom_half_lift",
            "lift_vs_random_top_k",
            "gp_lift_vs_random",
            "pool_enrichment",
            "hole_capture_rate",
            "hole_capture_lift_vs_chance",
            "p99_mae_pct",
            "p99_mae_delta",
            "num_ingest_ok",
            "num_selected",
            "n",
            "n_pre",
            "n_post",
            "verdict",
        ):
            if k in ev and ev[k] is not None:
                bits.append(f"{k}={_fmt(ev[k]) if not isinstance(ev[k], str) else ev[k]}")
        if c.get("note"):
            bits.append(c["note"])
        evid_s = "; ".join(bits) if bits else "—"
        lines.append(f"| {c['status']} | `{c['id']}` — {c['statement']} | {evid_s} |")

    lines.extend(["", "---", "", "## Key numbers (this cycle)", ""])
    imp = kn.get("improvement") or {}
    lines.append(
        f"- Pre p99 MAE: {_fmt(kn.get('pre_p99_mae'))} → Post: {_fmt(kn.get('post_p99_mae'))} "
        f"(Δ {_fmt(imp.get('p99_mae_delta'))}, {_fmt(imp.get('p99_mae_pct'), digits=1)}%)"
    )
    rank = kn.get("acquisition_rank") or {}
    if rank.get("available"):
        lines.append(
            f"- Acquisition Spearman(badness, p99_err): {_fmt(rank.get('spearman_badness_vs_p99_err'))} "
            f"(n={rank.get('n')}, ranking_ok={rank.get('ranking_ok')})"
        )
    else:
        lines.append("- Acquisition rank quality: unavailable")
    ctx = kn.get("cycle_context") or {}
    lines.append(
        f"- Pool → ECAD: candidates={ctx.get('num_candidates')} selected={ctx.get('num_selected')} "
        f"ingest_ok={ctx.get('num_ingest_ok')}/{ctx.get('num_ingest_total')}"
    )
    if ledger.get("acquisition_ab_path"):
        lines.append(f"- Acquisition A/B artifact: `{ledger['acquisition_ab_path']}`")

    lines.extend([
        "",
        "---",
        "",
        "## Artifacts",
        "",
        f"- `{DECISION_LEDGER}` (machine-readable claims)",
        f"- `{DECISION_REPORT_MD}` (this file)",
        f"- `iter_{int(ledger.get('iteration', 0)):04d}/{CYCLE_SUMMARY}`",
        f"- `iter_{int(ledger.get('iteration', 0)):04d}/{ACQUISITION_RANK}`",
        f"- `iter_{int(ledger.get('iteration', 0)):04d}/CYCLE_EVAL_REPORT.md`",
        "",
    ])
    return "\n".join(lines)


def write_decision_report(
    cfg: dict,
    groot: Path,
    iteration: int,
    *,
    summary: dict[str, Any] | None = None,
) -> Path:
    """Write decision_ledger.json + DECISION_REPORT.md under the run dir (and iter copy)."""
    eval_opts = cfg.get("evaluation") or {}
    if not eval_opts.get("write_decision_report", True):
        return run_dir(cfg, groot) / DECISION_REPORT_MD

    ledger = build_decision_ledger(cfg, groot, iteration, summary=summary)
    rdir = run_dir(cfg, groot)
    it_dir = iteration_dir(cfg, iteration, groot)

    save_json(rdir / DECISION_LEDGER, ledger)
    save_json(it_dir / DECISION_LEDGER, ledger)

    md = render_decision_markdown(ledger)
    out = rdir / DECISION_REPORT_MD
    out.write_text(md, encoding="utf-8")
    (it_dir / DECISION_REPORT_MD).write_text(md, encoding="utf-8")
    latest = rdir / f"LATEST_DECISION_REPORT_iter{iteration:04d}.md"
    latest.write_text(md, encoding="utf-8")
    return out


def cmd_write_decision_report(cfg: dict, groot: Path, iteration: int) -> Path:
    path = write_decision_report(cfg, groot, iteration)
    ledger = load_json(run_dir(cfg, groot) / DECISION_LEDGER)
    rollup = ledger.get("rollup") or {}
    print(f"\n=== Decision report (iter {iteration}) ===", flush=True)
    print(f"  overall: {rollup.get('overall')}", flush=True)
    print(f"  {rollup.get('summary')}", flush=True)
    for c in ledger.get("claims") or []:
        print(f"  [{c['status']}] {c['id']}", flush=True)
    print(f"  → {path}", flush=True)
    return path


def register_acquisition_ab(
    cfg: dict,
    groot: Path,
    ab_report: dict[str, Any],
    *,
    src_path: Path | None = None,
) -> Path:
    """Copy A/B JSON into the AL run dir so the next decision report can link it."""
    rdir = run_dir(cfg, groot)
    dest = rdir / "LATEST_acquisition_ab.json"
    payload = dict(ab_report)
    if src_path is not None:
        payload["source_path"] = str(src_path)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dest
