"""Active-learning orchestrator — full generate→simulate→ingest→normalize loop.

Run:
    Prefer ``python pipelines/active_learning/run.py`` (CONFIG at top)."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from active_learning_pi.al.acquisition import add_badness_scores, select_worst_for_simulation
from active_learning_pi.al.candidates import generate_candidates
from active_learning_pi.al.config import load_config, load_json, save_json
from active_learning_pi.al.ecadstar import run_ecadstar_batch
from active_learning_pi.al.evaluate_off_anchor import evaluate_labels_vs_predictions
from active_learning_pi.al.evaluate_cycle import (
    cmd_evaluate_pre_finetune,
    cmd_post_finetune_eval,
)
from active_learning_pi.al.decision_report import cmd_write_decision_report
from active_learning_pi.al.evaluate_report import cmd_write_cycle_report
from active_learning_pi.al.ingest_labels import ingest_simulation_outputs
from active_learning_pi.al.inference_pool import predict_candidates
from active_learning_pi.al.k_config import (
    candidates_per_k,
    resolve_k_values,
    stratify_ecad_by_k,
    total_ecad_batch_size,
    use_per_k_pools,
    worst_per_k,
)
from active_learning_pi.al.per_k_acquire import run_per_k_acquire
from active_learning_pi.al.paths import iteration_dir, repo_root, run_dir
from active_learning_pi.al.build_overlay import build_overlay_from_iterations
from active_learning_pi.al.normalize_labels import normalize_iteration_labels
from active_learning_pi.al.peb_batch import build_peb_from_selection

SELECTED_JSON = "selected_for_simulation.json"
PEB_NAME = "batch_simulate_once.peb"


def _state_path(cfg: dict, groot: Path) -> Path:
    return run_dir(cfg, groot) / "state.json"


def current_iteration(cfg: dict, groot: Path) -> int:
    p = _state_path(cfg, groot)
    if p.is_file():
        return int(load_json(p).get("iteration", 0))
    return 0


def bump_iteration(cfg: dict, groot: Path) -> int:
    p = _state_path(cfg, groot)
    n = current_iteration(cfg, groot) + 1
    save_json(p, {"iteration": n, "run_name": cfg["run_name"]})
    return n


def _peb_path(it_dir: Path) -> Path:
    return it_dir / PEB_NAME


def cmd_generate(cfg: dict, groot: Path, iteration: int) -> list[dict]:
    if use_per_k_pools(cfg):
        summary = run_per_k_acquire(cfg, groot, iteration)
        it_dir = iteration_dir(cfg, iteration, groot)
        return load_json(it_dir / "candidates.json")

    it_dir = iteration_dir(cfg, iteration, groot)
    k_values = resolve_k_values(cfg)
    print(f"\n=== Generate candidates (iter {iteration}) ===")
    cands = generate_candidates(
        num_candidates=int(cfg.get("num_candidates", 400)),
        mhz_grid=[float(x) for x in cfg["mhz_grid"]],
        k_values=k_values,
        seed=int(cfg["candidate_seed"]) + iteration,
        mhz_priority=[float(x) for x in cfg.get("mhz_priority", [])],
    )
    payload = [c.to_dict() for c in cands]
    save_json(it_dir / "candidates.json", payload)
    print(f"  Saved {len(payload)} candidates → {it_dir / 'candidates.json'}")
    return payload


def cmd_infer(cfg: dict, groot: Path, iteration: int) -> list[dict[str, Any]]:
    if use_per_k_pools(cfg):
        it_dir = iteration_dir(cfg, iteration, groot)
        path = it_dir / "scored_candidates.json"
        if not path.is_file():
            run_per_k_acquire(cfg, groot, iteration)
        return load_json(path)

    it_dir = iteration_dir(cfg, iteration, groot)
    cand_path = it_dir / "candidates.json"
    if not cand_path.is_file():
        raise FileNotFoundError(f"Run generate first — missing {cand_path}")

    from active_learning_pi.al.candidates import Candidate

    print(f"\n=== Infer + score all candidates (iter {iteration}) ===")
    raw = load_json(cand_path)
    cands = [Candidate(**{k: c[k] for k in ("candidate_id", "occupancy", "mhz", "k")}) for c in raw]

    scored = predict_candidates(cfg, cands, groot)
    scored = add_badness_scores(scored)
    save_json(it_dir / "scored_candidates.json", scored)
    print(f"  Scored {len(scored)} candidates → {it_dir / 'scored_candidates.json'}")
    return scored


def cmd_select_bad(cfg: dict, groot: Path, iteration: int) -> list[dict[str, Any]]:
    if use_per_k_pools(cfg):
        it_dir = iteration_dir(cfg, iteration, groot)
        path = it_dir / SELECTED_JSON
        if not path.is_file():
            run_per_k_acquire(cfg, groot, iteration)
        return load_json(path)

    it_dir = iteration_dir(cfg, iteration, groot)
    scored_path = it_dir / "scored_candidates.json"
    if not scored_path.is_file():
        raise FileNotFoundError(f"Run infer first — missing {scored_path}")

    print(f"\n=== Select worst candidates for ONE ECADStar batch (iter {iteration}) ===")
    scored = load_json(scored_path)
    sim_size = int(cfg.get("simulate_batch_size", cfg.get("batch_size", 8)))
    selected = select_worst_for_simulation(
        scored,
        simulate_batch_size=sim_size,
        min_uncertainty=cfg.get("min_uncertainty"),
        min_uncertainty_percentile=float(cfg.get("min_uncertainty_percentile", 0.0)),
        stratify_by_k=stratify_ecad_by_k(cfg),
    )
    save_json(it_dir / SELECTED_JSON, selected)

    with (it_dir / "selected_for_simulation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "candidate_id", "mhz", "k", "badness", "quality_score",
                "uncertainty", "pred_p99_mean", "selected_reason",
            ],
        )
        w.writeheader()
        for r in selected:
            w.writerow({k: r.get(k) for k in w.fieldnames})

    skipped = len(scored) - len(selected)
    print(f"  Worst (simulate) : {len(selected)} / {len(scored)}")
    print(f"  Skipped (good)   : {skipped}")
    if selected:
        print(f"  Badness range    : {selected[-1]['badness']:.6f} … {selected[0]['badness']:.6f}")
        from collections import Counter
        k_sel = Counter(int(s["k"]) for s in selected)
        print(f"  Selected K dist  : {dict(sorted(k_sel.items()))}", flush=True)
    return selected


def cmd_build_peb(cfg: dict, groot: Path, iteration: int) -> Path:
    it_dir = iteration_dir(cfg, iteration, groot)
    sel_path = it_dir / SELECTED_JSON
    if not sel_path.is_file():
        raise FileNotFoundError(f"Run select-bad first — missing {sel_path}")

    selected = load_json(sel_path)
    if not selected:
        raise RuntimeError("No candidates selected for simulation (all passed threshold?)")

    print(f"\n=== Build single PEB ({len(selected)} worst cases) ===")
    peb_path = _peb_path(it_dir)
    build_peb_from_selection(
        selected,
        peb_path,
        powerbus=str(cfg["powerbus"]),
        heatmap_only=bool(cfg.get("heatmap_only_peb", True)),
        components=str(cfg.get("peb_components", "IC1_Port1,IC2_Port2")),
        groot=groot,
    )
    print(f"  PEB → {peb_path}")
    return peb_path


def cmd_simulate(cfg: dict, groot: Path, iteration: int) -> int:
    it_dir = iteration_dir(cfg, iteration, groot)
    peb = _peb_path(it_dir)
    if not peb.is_file():
        raise FileNotFoundError(f"Run build-peb first — missing {peb}")
    selected = load_json(it_dir / SELECTED_JSON)
    print(f"\n=== Simulate ECADSTAR once (iter {iteration}) ===")
    print(f"  Selected layouts: {len(selected)}")
    return run_ecadstar_batch(cfg, peb, groot, pi_count=len(selected))


def cmd_ingest(cfg: dict, groot: Path, iteration: int) -> Path:
    it_dir = iteration_dir(cfg, iteration, groot)
    selected = load_json(it_dir / SELECTED_JSON)
    labels_dir = it_dir / "labels"
    print(f"\n=== Ingest labels (iter {iteration}) ===")
    manifest = ingest_simulation_outputs(cfg, selected, labels_dir, groot)
    save_json(it_dir / "ingest_manifest.json", manifest)
    ok = sum(1 for m in manifest if m.get("ingest_ok"))
    print(f"  Ingested {ok}/{len(manifest)} heatmaps → {labels_dir}")
    return labels_dir


def cmd_normalize(cfg: dict, groot: Path, iteration: int) -> dict[str, Any]:
    it_dir = iteration_dir(cfg, iteration, groot)
    print(f"\n=== Normalize labels (iter {iteration}) ===")
    return normalize_iteration_labels(cfg, it_dir, groot)


def cmd_evaluate(cfg: dict, groot: Path, iteration: int) -> dict[str, Any]:
    return cmd_evaluate_pre_finetune(cfg, groot, iteration)


def cmd_build_overlay(cfg: dict, groot: Path) -> dict[str, Any]:
    print(f"\n=== Build AL overlay dataset (Option B) ===")
    report = build_overlay_from_iterations(cfg, groot)
    print(
        f"  overlay: {report.get('overlay_root')}  added={report.get('added')}  "
        f"skipped={report.get('skipped_existing')}"
    )
    if report.get("errors"):
        for err in report["errors"][:8]:
            print(f"  error: {err}")
    return report


def cmd_finetune(cfg: dict, groot: Path) -> int:
    from active_learning_pi.al.finetune_run import finetune_env, prepare_line_buffered_logging, resolve_checkpoint_path

    prepare_line_buffered_logging()
    ft = cfg.get("finetune", {})
    if not ft.get("enabled"):
        print("\n=== Fine-tune ===")
        print("  finetune.enabled is false — enable in AL config or run:")
        print("    python pipelines/active_learning/finetune_exp058.py")
        return 1

    from active_learning_pi.al.build_overlay import build_overlay_from_iterations
    from src_vae.others.multifreq_layout_store import load_manifest_rows

    overlay_rel = ft.get("overlay_data_dir") or cfg.get("overlay_data_dir")
    overlay_root = groot / overlay_rel if overlay_rel else None
    n_overlay = len(load_manifest_rows(overlay_root)) if overlay_root and overlay_root.is_dir() else 0

    if n_overlay == 0:
        print("\n=== Build AL overlay (required for fine-tune) ===")
        report = build_overlay_from_iterations(cfg, groot)
        n_overlay = int(report.get("added", 0)) + int(report.get("skipped_existing", 0))
        if n_overlay == 0:
            print("  No AL overlay samples — run ingest first or use --train-only.")
            return 1

    exp_rel = ft["experiment_dir"]
    train_script = groot / exp_rel / "codes" / "train_vae_simple.py"
    if not train_script.is_file():
        print(f"Missing training script: {train_script}")
        return 1

    import subprocess
    import sys

    ckpt = resolve_checkpoint_path(cfg, groot)
    print(f"\n=== Fine-tune {Path(exp_rel).name} (Option B / asymmetric) ===", flush=True)
    print(f"  checkpoint: {ckpt}", flush=True)
    print(f"  overlay samples: {n_overlay}", flush=True)
    print(
        f"  layout_p={ft.get('layout_train_prob', '?')}  "
        f"occ_only={ft.get('occ_only_encode_prob', '?')}",
        flush=True,
    )

    env = finetune_env(cfg, groot, use_overlay=True)
    return int(subprocess.call([sys.executable, str(train_script)], cwd=str(groot), env=env))


def _resolve_occ_warmup_script(cfg: dict, groot: Path) -> Path:
    """Prefer exp-specific warmup script; fall back to exp058 then exp057."""
    exp = str(cfg.get("experiment_dir", ""))
    candidates = []
    if "exp058" in exp:
        candidates.append(groot / "pipelines/active_learning/finetune_occ_only_exp058.py")
    if "exp057" in exp:
        candidates.append(groot / "pipelines/active_learning/finetune_occ_only_exp057.py")
    candidates.extend(
        [
            groot / "pipelines/active_learning/finetune_occ_only_exp058.py",
            groot / "pipelines/active_learning/finetune_occ_only_exp057.py",
        ]
    )
    for script in candidates:
        if script.is_file():
            return script
    return candidates[0]


def cmd_occ_only_warmup(cfg: dict, groot: Path) -> int:
    """Pre-AL occupancy-only fine-tune (no overlay) — run before candidate scoring."""
    import subprocess
    import sys

    script = _resolve_occ_warmup_script(cfg, groot)
    if not script.is_file():
        print(f"Missing: {script}")
        return 1
    print(f"\n=== Pre-AL occupancy-only warm-up ===\n  script: {script}", flush=True)
    return int(subprocess.call([sys.executable, str(script)], cwd=str(groot)))


def cmd_finetune_hint(cfg: dict, groot: Path) -> None:
    ft = cfg.get("finetune", {})
    exp_name = Path(str(ft.get("experiment_dir") or cfg.get("experiment_dir") or "exp058")).name
    ft_script = (
        "finetune_exp058.py" if "exp058" in exp_name else "finetune_exp057.py"
    )
    if not ft.get("enabled"):
        print("\n=== Fine-tune ===")
        print("  finetune.enabled is false.")
        print("  After AL ingest, run:")
        print(f"    python pipelines/active_learning/{ft_script}")
        print("  Or set COMMAND=build-overlay then COMMAND=finetune in pipelines/active_learning/run.py")
        return
    print(f"\n=== Fine-tune (Option B) ===")
    print(f"  1) python pipelines/active_learning/run.py   # COMMAND=build-overlay")
    print(f"  2) python pipelines/active_learning/{ft_script}")
    print(f"     or COMMAND=finetune in pipelines/active_learning/run.py")
    print(f"  config: {ft.get('config_path')}")
    print(f"  overlay: {ft.get('overlay_data_dir', cfg.get('overlay_data_dir'))}")


def run_cycle(
    cfg: dict,
    groot: Path,
    *,
    do_simulate: bool,
    do_ingest: bool,
    run_finetune: bool = False,
    run_post_eval: bool | None = None,
) -> int:
    """
    Active-learning cycle:
      1) generate candidates
      2) infer + score (MC uncertainty)
      3) select worst → PEB
      4) ECADStar batch
      5) ingest + normalize + pre-finetune evaluate (numericals → decision ledger)
      6) [optional] build overlay + fine-tune
      7) [optional] post-finetune full evaluation + DECISION_REPORT
    """
    ft = cfg.get("finetune", {})
    eval_opts = cfg.get("evaluation") or {}
    # Numerical evaluation is always required after ingest; only post-FT can be toggled.
    if run_post_eval is None:
        run_post_eval = bool(eval_opts.get("post_finetune", True))
    n_steps = 8 if run_finetune and ft.get("enabled") and run_post_eval else (
        7 if run_finetune and ft.get("enabled") else 6
    )

    it = bump_iteration(cfg, groot)
    print(f"\n{'='*60}\n  AL iteration {it}\n{'='*60}")

    print(f"\n[1/{n_steps}] Per-K acquire (generate + infer + select)")
    if use_per_k_pools(cfg):
        run_per_k_acquire(cfg, groot, it)
    else:
        cmd_generate(cfg, groot, it)
        print(f"\n[2/{n_steps}] Infer + score (uncertainty)")
        cmd_infer(cfg, groot, it)
        print(f"\n[3/{n_steps}] Select worst for ECAD")
        cmd_select_bad(cfg, groot, it)
    print(f"\n[4/{n_steps}] Build PEB")
    cmd_build_peb(cfg, groot, it)

    rc = 0
    if do_simulate:
        print(f"\n[5/{n_steps}] ECADStar simulate")
        rc = cmd_simulate(cfg, groot, it)
        if rc != 0:
            return rc
    else:
        print(f"\n[5/{n_steps}] ECADStar simulate — SKIPPED")

    step = 6
    if do_ingest and rc == 0:
        print(f"\n[{step}/{n_steps}] Ingest + normalize + pre-finetune evaluate")
        cmd_ingest(cfg, groot, it)
        cmd_normalize(cfg, groot, it)
        cmd_evaluate_pre_finetune(cfg, groot, it)
        step += 1
    elif not do_ingest:
        print(f"\n[{step}/{n_steps}] Ingest — SKIPPED")
        step += 1

    if run_finetune and do_ingest and rc == 0 and ft.get("enabled"):
        print(f"\n[{step}/{n_steps}] Build overlay + fine-tune")
        cmd_build_overlay(cfg, groot)
        rc = cmd_finetune(cfg, groot)
        step += 1
        if rc == 0 and run_post_eval:
            print(f"\n[{step}/{n_steps}] Post-finetune eval + decision report")
            cmd_post_finetune_eval(cfg, groot, it)
        return rc

    if run_finetune and ft.get("enabled"):
        cmd_finetune_hint(cfg, groot)
    else:
        print(f"\n[{step}/{n_steps}] Fine-tune — skipped (use COMMAND=full or finetune.run_after_cycle=true)")
    return rc


def run_full_cycle(
    cfg: dict,
    groot: Path,
    *,
    skip_simulate: bool = False,
    skip_ingest: bool = False,
) -> int:
    """End-to-end: cycle + overlay + fine-tune (single command)."""
    return run_cycle(
        cfg,
        groot,
        do_simulate=not skip_simulate,
        do_ingest=not skip_ingest,
        run_finetune=True,
    )


def main_from_config(
    *,
    command: str,
    config_path: str | Path | None = None,
    iteration: int | None = None,
    skip_simulate: bool = False,
    skip_ingest: bool = False,
) -> int:
    """Run pipeline from CONFIG. Called by pipelines/active_learning/run.py."""
    cfg = load_config(config_path)
    groot = Path(cfg.get("repo_root") or cfg["gan_root"])

    def _it() -> int:
        return iteration if iteration is not None else current_iteration(cfg, groot)

    if command == "generate":
        it = bump_iteration(cfg, groot)
        cmd_generate(cfg, groot, it)
        return 0
    if command == "infer":
        cmd_infer(cfg, groot, _it())
        return 0
    if command == "select-bad":
        cmd_select_bad(cfg, groot, _it())
        return 0
    if command == "build-peb":
        cmd_build_peb(cfg, groot, _it())
        return 0
    if command == "simulate":
        return cmd_simulate(cfg, groot, _it())
    if command == "ingest":
        cmd_ingest(cfg, groot, _it())
        return 0
    if command == "normalize":
        cmd_normalize(cfg, groot, _it())
        return 0
    if command == "evaluate":
        cmd_evaluate(cfg, groot, _it())
        return 0
    if command == "evaluate-pre-finetune":
        cmd_evaluate_pre_finetune(cfg, groot, _it())
        return 0
    if command == "evaluate-post-finetune":
        cmd_post_finetune_eval(cfg, groot, _it())
        return 0
    if command == "evaluate-full":
        cmd_post_finetune_eval(cfg, groot, _it())
        return 0
    if command == "evaluate-report":
        cmd_write_cycle_report(cfg, groot, _it())
        return 0
    if command == "evaluate-decision":
        cmd_write_decision_report(cfg, groot, _it())
        return 0
    if command == "build-overlay":
        cmd_build_overlay(cfg, groot)
        return 0
    if command == "finetune":
        return cmd_finetune(cfg, groot)
    if command == "occ-warmup":
        return cmd_occ_only_warmup(cfg, groot)
    if command == "finetune-hint":
        cmd_finetune_hint(cfg, groot)
        return 0
    if command in ("cycle", "iteration"):
        ft = cfg.get("finetune", {})
        auto_ft = bool(ft.get("run_after_cycle", False))
        return run_cycle(
            cfg,
            groot,
            do_simulate=not skip_simulate,
            do_ingest=not skip_ingest,
            run_finetune=auto_ft,
        )
    if command == "full":
        return run_full_cycle(
            cfg,
            groot,
            skip_simulate=skip_simulate,
            skip_ingest=skip_ingest,
        )
    raise ValueError(f"Unknown command: {command!r}")


if __name__ == "__main__":
    raise SystemExit(
        "Edit CONFIG in pipelines/active_learning/run.py and run:\n"
        "  python pipelines/active_learning/run.py"
    )
