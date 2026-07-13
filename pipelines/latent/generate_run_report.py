"""Run report generator — Markdown summary for a latent optimization run.

Run:
    python pipelines/latent/generate_run_report.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# =============================================================================
# CONFIGURATION — edit these before running: python pipelines/latent/generate_run_report.py
# =============================================================================

# Run folder (None = require path — set explicitly when running standalone)
RUN_DIR: str | Path | None = None  # e.g. data/latent_runs/exp038_true_multi/0

# =============================================================================


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_list(v: list, max_items: int = 10) -> str:
    if len(v) <= max_items:
        return str(v)
    return f"[{v[0]}, {v[1]}, …, {v[-2]}, {v[-1]}] ({len(v)} items)"


def _fmt_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.1f} s"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{int(m)}m {sec:.0f}s"
    h, m2 = divmod(m, 60)
    return f"{int(h)}h {int(m2)}m"


def _isnan(v: float) -> bool:
    return v != v  # works for float("nan")


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(run_folder: Path) -> str:
    cfg_path    = run_folder / "run_config.json"
    timing_path = run_folder / "timing_summary.json"

    if not cfg_path.exists():
        raise FileNotFoundError(f"run_config.json not found in {run_folder}")

    cfg    = json.loads(cfg_path.read_text(encoding="utf-8"))
    timing = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.exists() else {}

    run_name    = run_folder.name
    experiment  = cfg.get("experiment", "—")
    run_index   = cfg.get("run_index", timing.get("run_index", "—"))
    started     = timing.get("started", cfg.get("started", "—"))
    device      = cfg.get("device", timing.get("device", "—"))
    checkpoint  = cfg.get("checkpoint", "—")
    norm_path   = cfg.get("normalization_stats_path", "—")
    total_sec   = timing.get("seconds_total", None)
    total_str   = _fmt_seconds(total_sec) if total_sec else "—"

    k_list      = cfg.get("K_list", [])
    seeds       = cfg.get("seeds", [])
    batch_mode  = cfg.get("optimize_batch_per_k", False)
    seed_info   = cfg.get("seed_info", {})

    lines: list[str] = []
    a = lines.append

    # Header
    a(f"# Latent Optimisation Run Report")
    a(f"")
    a(f"**Experiment:** `{experiment}`  ")
    a(f"**Run:** `{run_name}` (index {run_index})  ")
    a(f"**Started:** {started}  ")
    a(f"**Device:** {device}  ")
    a(f"**Total runtime:** {total_str}  ")
    a(f"**Checkpoint:** `{checkpoint}`  ")
    a(f"**Norm stats:** `{norm_path}`  ")
    a(f"")
    if seed_info:
        a(f"**Seed mode:** `{seed_info.get('mode', '—')}`  ")
        if seed_info.get("master_seed") is not None:
            a(f"**Master seed:** {seed_info.get('master_seed')}  ")
        a(f"")

    # Model dims
    a(f"## Model")
    a(f"")
    a(f"| Parameter | Value |")
    a(f"|-----------|-------|")
    a(f"| `latent_dim` | {cfg.get('latent_dim', '—')} |")
    a(f"| `cond_dim` | {cfg.get('cond_dim', '—')} |")
    a(f"| `heatmap_private_dim` | {cfg.get('heatmap_private_dim', '—')} |")
    a(f"")

    # Search space
    a(f"## Search Space")
    a(f"")
    a(f"| Parameter | Value |")
    a(f"|-----------|-------|")
    a(f"| K values | `{_fmt_list(k_list)}` |")
    a(f"| Seeds | `{_fmt_list(seeds)}` |")
    a(f"| Mode | {'batch (joint + diversity penalty)' if batch_mode else 'independent per seed'} |")
    a(f"")

    # Optimisation hyperparameters
    a(f"## Optimisation Hyperparameters")
    a(f"")
    a(f"| Parameter | Value |")
    a(f"|-----------|-------|")
    a(f"| `num_steps` | {cfg.get('num_steps', '—')} |")
    a(f"| `lr` | {cfg.get('lr', '—')} |")
    a(f"| `grad_clip` | {cfg.get('grad_clip', '—')} |")
    a(f"| `init_shared_temp` | {cfg.get('init_shared_temp', '—')} |")
    a(f"| `loss_space` | `{cfg.get('loss_space', '—')}` |")
    a(f"| `objective_mode` | `{cfg.get('objective_mode', '—')}` |")
    a(f"")

    # Loss / regularisation
    a(f"## Loss & Regularisation Weights")
    a(f"")
    a(f"| Weight | Value |")
    a(f"|--------|-------|")
    a(f"| `z_l2_weight` | {cfg.get('z_l2_weight', '—')} |")
    a(f"| `z_prior_mode` | `{cfg.get('z_prior_mode', '—')}` |")

    bt = cfg.get("below_target", {})
    a(f"| `boundary_margin` | {bt.get('margin', '—')} |")
    a(f"| `gap_reward_weight` | {bt.get('gap_reward_weight', '—')} |")
    a(f"| `exceed_weight` | {bt.get('exceed_weight', '—')} |")
    a(f"| `exceed_power` | {bt.get('exceed_power', '—')} |")

    sr = cfg.get("shape_reg", {})
    a(f"| `shape_reg_weight` | {sr.get('weight', '—')} |")
    a(f"| `shape_min_std_dlog` | {sr.get('min_std_dlog', '—')} |")
    a(f"| `chan_rough_weight` | {cfg.get('chan_rough_weight', '—')} |")

    pb = cfg.get("posterior_boundary", {})
    a(f"| `posterior_boundary_weight` | {pb.get('weight', '—')} |")
    a(f"| `posterior_sigma_limit` | {pb.get('sigma_limit', '—')} |")

    div = cfg.get("diversity", {})
    a(f"| `diversity_weight` | {div.get('weight', '—')} |")
    a(f"| `occ_confidence_weight` | {div.get('occ_confidence_weight', '—')} |")
    a(f"")

    # Normalisation stats (exp038 run_config uses imp_log_norm)
    ns = cfg.get("imp_log_norm") or cfg.get("normalization_stats")
    if ns:
        a(f"## Normalisation Statistics (impedance log)")
        a(f"")
        a(f"| Stat | Value |")
        a(f"|------|-------|")
        for k, v in ns.items():
            a(f"| `{k}` | {v} |")
        a(f"")

    # Per-K timing
    per_k = timing.get("per_k", {})
    if per_k:
        a(f"## Per-K Timing")
        a(f"")
        a(f"| K | Seconds | Solution | max_ohm | winning_seed |")
        a(f"|---|---------|----------|---------|--------------|")
        for k_str, kdata in sorted(per_k.items(), key=lambda x: int(x[0])):
            secs = kdata.get("seconds", None)
            sec_str = _fmt_seconds(secs) if secs is not None else "—"
            has_sol = kdata.get("has_solution", kdata.get("feasible"))
            if has_sol is None and kdata.get("mode") == "batch":
                has_sol = kdata.get("feasible")
            sol_str = "yes" if has_sol else ("no" if has_sol is False else "—")
            max_ohm = kdata.get("max_imp_ohm", "—")
            if isinstance(max_ohm, float):
                max_ohm = f"{max_ohm:.4g}"
            seed = kdata.get("seed", "—")
            a(f"| {k_str} | {sec_str} | {sol_str} | {max_ohm} | {seed} |")
        a(f"")

    # No-solution summary (exp038: one best per K under K##/)
    no_sol_ks: list[int] = []
    for k_str, kdata in sorted(per_k.items(), key=lambda x: int(x[0])):
        if kdata.get("has_solution") is False:
            no_sol_ks.append(int(k_str))
        elif not (run_folder / f"K{int(k_str):02d}" / "best_metrics.json").exists():
            if not (run_folder / f"K{int(k_str):02d}" / "feasible_best_metrics.json").exists():
                if kdata.get("has_solution") is not True:
                    no_sol_ks.append(int(k_str))
    if no_sol_ks:
        a(f"## No Solution Found")
        a(f"")
        a(f"No feasible impedance solution was saved for K:")
        a(f"")
        a(f"| K |")
        a(f"|---|")
        for k_int in no_sol_ks:
            a(f"| {k_int} |")
        a(f"")
    else:
        a(f"## No Solution Found")
        a(f"")
        a(f"Every K in the run produced a saved `best_*` solution (or the run is incomplete).")
        a(f"")

    # -----------------------------------------------------------------------
    # Results summary — scan saved solution metrics
    # -----------------------------------------------------------------------
    a(f"## Results Summary")
    a(f"")

    def _metrics_file(k_dir: Path) -> Path | None:
        for name in ("best_metrics.json", "feasible_best_metrics.json"):
            p = k_dir / name
            if p.exists():
                return p
        return None

    result_rows: list[dict] = []
    for k_str in sorted(per_k.keys(), key=lambda x: int(x[0])):
        k_int = int(k_str)
        k_dir = run_folder / f"K{k_int:02d}"
        mf = _metrics_file(k_dir)
        if mf is None:
            result_rows.append({"K": k_int, "has_solution": False})
            continue
        m = json.loads(mf.read_text(encoding="utf-8"))
        result_rows.append({
            "K": k_int,
            "has_solution": True,
            "max_ohm": float(m.get("max_imp_ohm", float("nan"))),
            "best_score": float(m.get("best_score", m.get("best_total", float("nan")))),
            "best_step": int(m.get("best_step", -1)),
            "winning_seed": m.get("winning_seed", "—"),
        })

    if result_rows:
        a(f"| K | Solution | max_ohm | best_score | Step | Seed |")
        a(f"|---|----------|---------|------------|------|------|")
        for r in result_rows:
            if not r.get("has_solution"):
                a(f"| {r['K']} | no | — | — | — | — |")
                continue
            mo = f"{r['max_ohm']:.4g}" if not _isnan(r["max_ohm"]) else "—"
            bs = f"{r['best_score']:.4g}" if not _isnan(r["best_score"]) else "—"
            step = str(r["best_step"]) if r["best_step"] >= 0 else "—"
            a(f"| {r['K']} | yes | {mo} | {bs} | {step} | {r['winning_seed']} |")
        a(f"")

        total_k = len(result_rows)
        n_sol = sum(1 for r in result_rows if r.get("has_solution"))
        a(f"**K with saved solution:** **{n_sol}/{total_k}**  ")
        a(f"")
    else:
        a(f"No solution metrics found (run may not have completed).")
        a(f"")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def write_report(run_folder: Path) -> Path:
    report_text = build_report(run_folder)
    out_path    = run_folder / "run_report.md"
    out_path.write_text(report_text, encoding="utf-8")
    return out_path


def main() -> None:
    if RUN_DIR is None:
        raise SystemExit(
            "Set RUN_DIR in CONFIGURATION block, e.g. RUN_DIR = 'data/latent_runs/exp038_true_multi/0'"
        )
    run_folder = Path(RUN_DIR)
    if not run_folder.is_absolute():
        from repo_paths import REPO_ROOT
        run_folder = (REPO_ROOT / run_folder).resolve()
    if not run_folder.is_dir():
        print(f"Error: {run_folder} is not a directory")
        sys.exit(1)
    out = write_report(run_folder)
    print(f"Report written → {out}")


if __name__ == "__main__":
    main()
