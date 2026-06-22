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
from active_learning_pi.al.ingest_labels import ingest_simulation_outputs
from active_learning_pi.al.inference_pool import predict_candidates
from active_learning_pi.al.paths import gan_root, iteration_dir, run_dir
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
    it_dir = iteration_dir(cfg, iteration, groot)
    print(f"\n=== Generate candidates (iter {iteration}) ===")
    cands = generate_candidates(
        num_candidates=int(cfg["num_candidates"]),
        mhz_grid=[float(x) for x in cfg["mhz_grid"]],
        fixed_k=int(cfg["fixed_k"]),
        seed=int(cfg["candidate_seed"]) + iteration,
        mhz_priority=[float(x) for x in cfg.get("mhz_priority", [])],
    )
    payload = [c.to_dict() for c in cands]
    save_json(it_dir / "candidates.json", payload)
    print(f"  Saved {len(payload)} candidates → {it_dir / 'candidates.json'}")
    return payload


def cmd_infer(cfg: dict, groot: Path, iteration: int) -> list[dict[str, Any]]:
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
    print(f"\n=== Simulate ECADSTAR once (iter {iteration}) ===")
    return run_ecadstar_batch(cfg, peb, groot)


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
    it_dir = iteration_dir(cfg, iteration, groot)
    manifest = load_json(it_dir / "ingest_manifest.json")
    scored = load_json(it_dir / "scored_candidates.json")
    report = evaluate_labels_vs_predictions(
        manifest,
        scored,
        off_anchor_mhz=[float(x) for x in cfg.get("eval_off_anchor_mhz", [80, 250])],
    )
    save_json(it_dir / "eval_off_anchor.json", report)
    print(f"\n=== Evaluate off-anchor (iter {iteration}) ===")
    print(f"  n={report['n']}  p99_mae={report.get('p99_mae', 'n/a')}")
    return report


def cmd_finetune_hint(cfg: dict, groot: Path) -> None:
    ft = cfg.get("finetune", {})
    if not ft.get("enabled"):
        print("\n=== Fine-tune ===")
        print("  finetune.enabled is false.")
        print("  Merge runs/.../iter_*/dataset_norm/ into your training set, then:")
        print(f"    python {ft.get('experiment_dir', 'experiments/exp039_improved_heatmap')}/codes/train_vae_simple.py")
        return
    exp = groot / ft["experiment_dir"]
    print(f"\n=== Fine-tune ===")
    print(f"  cd {groot}")
    print(f"  python {exp}/codes/train_vae_simple.py  # config: {ft.get('config_path')}")


def run_cycle(
    cfg: dict,
    groot: Path,
    *,
    do_simulate: bool,
    do_ingest: bool,
) -> int:
    """
    Full active-learning cycle:
      1) generate all candidates
      2) infer + score each (cheap, no ECADStar)
      3) select only worst → one PEB
      4) one ECADStar batch
      5) ingest + evaluate
      6) fine-tune hint
    """
    it = bump_iteration(cfg, groot)
    cmd_generate(cfg, groot, it)
    cmd_infer(cfg, groot, it)
    cmd_select_bad(cfg, groot, it)
    cmd_build_peb(cfg, groot, it)

    rc = 0
    if do_simulate:
        rc = cmd_simulate(cfg, groot, it)
        if rc != 0:
            return rc
    if do_ingest and rc == 0:
        cmd_ingest(cfg, groot, it)
        cmd_normalize(cfg, groot, it)
        cmd_evaluate(cfg, groot, it)
    cmd_finetune_hint(cfg, groot)
    return rc


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
    groot = Path(cfg["gan_root"])

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
    if command == "finetune-hint":
        cmd_finetune_hint(cfg, groot)
        return 0
    if command in ("cycle", "iteration"):
        return run_cycle(
            cfg,
            groot,
            do_simulate=not skip_simulate,
            do_ingest=not skip_ingest,
        )
    raise ValueError(f"Unknown command: {command!r}")


if __name__ == "__main__":
    raise SystemExit(
        "Edit CONFIG in pipelines/active_learning/run.py and run:\n"
        "  python pipelines/active_learning/run.py"
    )
