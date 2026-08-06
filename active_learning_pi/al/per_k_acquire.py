"""Per-K candidate pools: N candidates and worst-M selection for each K value."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

from active_learning_pi.al.acquisition import add_badness_scores, select_worst_for_simulation
from active_learning_pi.al.candidates import Candidate, generate_candidates
from active_learning_pi.al.config import load_json, save_json
from active_learning_pi.al.inference_pool import predict_candidates
from active_learning_pi.al.k_config import candidates_per_k, resolve_k_values, resolve_worst_per_k_values, worst_per_k
from active_learning_pi.al.mhz_strata import resolve_mhz_strata_per_k, resolve_worst_mhz_strata_per_k
from active_learning_pi.al.paths import iteration_dir

SELECTED_JSON = "selected_for_simulation.json"
K_SWEEP_SUMMARY = "k_sweep_summary.json"


def _write_selected_csv(it_dir: Path, selected: list[dict[str, Any]]) -> None:
    with (it_dir / "selected_for_simulation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "candidate_id", "mhz", "k", "badness", "quality_score",
                "uncertainty", "heatmap_mc_mse", "decoder_mc_rce",
                "epistemic_score", "disagreement_score", "prior_tension_score",
                "peak_bias_score", "p99_var", "p99_mc_range", "peak_loc_spread_px",
                "pred_p99_mean", "selected_reason",
            ],
        )
        w.writeheader()
        for r in selected:
            w.writerow({k: r.get(k) for k in w.fieldnames})


def run_per_k_acquire(cfg: dict, groot: Path, iteration: int) -> dict[str, Any]:
    """
  For each K in ``k_min..k_max``:
    - generate ``candidates_per_k`` layouts at that K
    - MC infer + score
    - select ``worst_per_k`` for ECAD

  Writes combined ``candidates.json``, ``scored_candidates.json``, ``selected_for_simulation.json``.
  """
    it_dir = iteration_dir(cfg, iteration, groot)
    k_values = resolve_k_values(cfg)
    pool = candidates_per_k(cfg)
    worst_dist = resolve_worst_per_k_values(cfg)
    worst_n_default = worst_per_k(cfg)
    mhz_strata = resolve_mhz_strata_per_k(cfg)
    worst_mhz_default = resolve_worst_mhz_strata_per_k(cfg)
    explore_n = int(cfg.get("explore_candidates_per_k", 0) or 0)
    explore_mhz_grid = cfg.get("explore_mhz_grid")
    explore_mhz_min = cfg.get("explore_mhz_min")
    explore_mhz_max = cfg.get("explore_mhz_max")
    explore_mhz_quantize = float(cfg.get("explore_mhz_quantize", 0.0) or 0.0)
    explore_mhz_band_edges = cfg.get("explore_mhz_band_edges")
    seed_base = int(cfg["candidate_seed"]) + iteration * 10_000

    print(f"\n=== Per-K acquire (iter {iteration}) ===", flush=True)
    acq_mode = str(cfg.get("acquisition_mode", "mc")).strip().lower()
    print(f"  acquisition_mode: {acq_mode}", flush=True)
    # Let GP scorer write metadata under this iteration dir; clear cached fit
    cfg = {**cfg, "_iteration": int(iteration)}
    cfg.pop("_gp_error_artifact", None)
    cfg.pop("_gp_error_engine", None)
    cfg.pop("_gp_error_exp_yaml", None)
    print(f"  K values: {k_values}", flush=True)
    if worst_dist:
        print(f"  Per K: {pool} candidates → worst varies (total_ecad_per_cycle={sum(worst_dist.values())})", flush=True)
        print(f"  Worst-per-K: {worst_dist}", flush=True)
    else:
        print(f"  Per K: {pool} candidates → worst {worst_n_default}", flush=True)
    if mhz_strata:
        print(f"  MHz strata (per K): {dict(sorted(mhz_strata.items()))}", flush=True)
    if worst_mhz_default:
        print(f"  ECAD MHz quotas (per K): {dict(sorted(worst_mhz_default.items()))}", flush=True)
    if explore_n > 0:
        if explore_mhz_grid:
            print(f"  Exploration: {explore_n} / K from explore_mhz_grid={explore_mhz_grid}", flush=True)
        else:
            print(
                f"  Exploration: {explore_n} / K from range {explore_mhz_min}..{explore_mhz_max} "
                f"(quantize={explore_mhz_quantize}, bands={explore_mhz_band_edges})",
                flush=True,
            )
    total_ecad = sum(worst_dist.values()) if worst_dist else len(k_values) * worst_n_default
    print(f"  Total pool: {len(k_values) * pool}  |  Total ECAD: {total_ecad}", flush=True)

    all_candidates: list[dict[str, Any]] = []
    all_scored: list[dict[str, Any]] = []
    all_selected: list[dict[str, Any]] = []
    k_rows: list[dict[str, Any]] = []

    for ki, k in enumerate(k_values):
        worst_n = int(worst_dist.get(int(k), worst_n_default) if worst_dist else worst_n_default)
        print(f"\n--- K={k} ({ki + 1}/{len(k_values)}) ---", flush=True)
        cands = generate_candidates(
            num_candidates=pool,
            mhz_grid=[float(x) for x in cfg["mhz_grid"]],
            k_values=[k],
            seed=seed_base + k,
            mhz_strata=mhz_strata,
            mhz_priority=None,
            explore_mhz_grid=[float(x) for x in explore_mhz_grid] if explore_mhz_grid else None,
            explore_n=explore_n,
            explore_mhz_min=float(explore_mhz_min) if explore_mhz_min is not None else None,
            explore_mhz_max=float(explore_mhz_max) if explore_mhz_max is not None else None,
            explore_mhz_quantize=explore_mhz_quantize,
            explore_mhz_band_edges=[float(x) for x in explore_mhz_band_edges] if explore_mhz_band_edges else None,
        )
        id_base = ki * pool
        for i, c in enumerate(cands):
            c.candidate_id = id_base + i

        cand_dicts = [c.to_dict() for c in cands]
        all_candidates.extend(cand_dicts)

        cands_objs = [
            Candidate(**{key: d[key] for key in ("candidate_id", "occupancy", "mhz", "k")})
            for d in cand_dicts
        ]
        scored = add_badness_scores(predict_candidates(cfg, cands_objs, groot))
        worst_mhz = worst_mhz_default
        if (not cfg.get("worst_mhz_strata_per_k")) and mhz_strata:
            # Auto-scale MHz quotas to this K's worst_n
            from active_learning_pi.al.mhz_strata import scale_worst_mhz_strata_from_pool

            worst_mhz = scale_worst_mhz_strata_from_pool(mhz_strata, worst_n)
        selected = select_worst_for_simulation(
            scored,
            simulate_batch_size=worst_n,
            min_uncertainty=cfg.get("min_uncertainty"),
            min_uncertainty_percentile=float(cfg.get("min_uncertainty_percentile", 0.0)),
            stratify_by_k=False,
            mhz_quotas=worst_mhz,
        )
        all_scored.extend(scored)
        all_selected.extend(selected)

        bad_min = float(selected[-1]["badness"]) if selected else None
        bad_max = float(selected[0]["badness"]) if selected else None
        cand_mhz = dict(sorted(Counter(round(c.mhz, 6) for c in cands).items()))
        sel_mhz = dict(sorted(Counter(round(float(s["mhz"]), 6) for s in selected).items()))
        k_rows.append({
            "k": k,
            "candidates": pool,
            "scored": len(scored),
            "selected": len(selected),
            "badness_min": bad_min,
            "badness_max": bad_max,
            "mhz_candidate_distribution": cand_mhz,
            "mhz_selected_distribution": sel_mhz,
        })
        print(
            f"  K={k}: scored {len(scored)}  selected {len(selected)}  "
            f"badness {bad_min:.4f}…{bad_max:.4f}" if selected else f"  K={k}: no selection",
            flush=True,
        )
        print(f"  MHz selected: {sel_mhz}", flush=True)

    save_json(it_dir / "candidates.json", all_candidates)
    save_json(it_dir / "scored_candidates.json", all_scored)
    save_json(it_dir / SELECTED_JSON, all_selected)
    _write_selected_csv(it_dir, all_selected)

    sel_k = dict(sorted(Counter(int(s["k"]) for s in all_selected).items()))
    sel_mhz_all = dict(sorted(Counter(round(float(s["mhz"]), 6) for s in all_selected).items()))
    summary = {
        "iteration": iteration,
        "acquisition_mode": acq_mode,
        "k_values": k_values,
        "candidates_per_k": pool,
        "worst_per_k": worst_n_default,
        "worst_per_k_distribution": worst_dist,
        "mhz_strata_per_k": mhz_strata,
        "worst_mhz_strata_per_k": worst_mhz_default,
        "total_candidates": len(all_candidates),
        "total_selected": len(all_selected),
        "selected_k_distribution": sel_k,
        "selected_mhz_distribution": sel_mhz_all,
        "per_k": k_rows,
    }
    save_json(it_dir / K_SWEEP_SUMMARY, summary)

    print(f"\n  Combined: {len(all_candidates)} candidates, {len(all_selected)} selected for ECAD", flush=True)
    print(f"  Selected K dist: {sel_k}", flush=True)
    print(f"  Selected MHz dist: {sel_mhz_all}", flush=True)
    print(f"  Summary → {it_dir / K_SWEEP_SUMMARY}", flush=True)
    return summary


def load_k_sweep_summary(cfg: dict, iteration: int, groot: Path) -> dict[str, Any] | None:
    p = iteration_dir(cfg, iteration, groot) / K_SWEEP_SUMMARY
    return load_json(p) if p.is_file() else None
