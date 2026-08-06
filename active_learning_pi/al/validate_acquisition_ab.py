"""Equal-budget acquisition A/B: GP-UCB (+novelty) vs random vs optional MC.

Falsifies the claim that residual-GP acquisition selects higher *true* VAE error
than random (and MC) at the same simulation budget — **without new ECAD**.

Method
------
1. Collect labeled residuals on the production ``pred`` path (occ → decode → re-encode).
2. Hold out a test slice; fit GP (+ novelty bank) on the remainder.
3. Rank the holdout by: GP acquisition, random, and optionally MC ``auto_bad``.
4. Report Spearman(score, true y) and mean true y in the top-k (equal budget).

Pass criterion (thesis):
  mean_true_y(top-k GP) > mean_true_y(top-k random)  AND  Spearman(GP) > 0

Run:
    python active_learning_pi/al/validate_acquisition_ab.py

Agent notes:
    - What: cheap labeled holdout A/B for GP vs random (optional MC).
    - Usage: edit CONFIG below, then run. Writes JSON under ``OUT_DIR``.
    - Config keys: EXPERIMENT, N_SAMPLES, BUDGET_K, INCLUDE_MC, GP_* .
"""
from __future__ import annotations

from repo_paths import setup_path

setup_path()

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

# ── CONFIGURATION — edit these before running ─────────────────────────────────

EXPERIMENT = "experiments/exp059_capacity_freq"
CHECKPOINT = None  # None → <exp>/checkpoints/last_model.pt
DATA_DIR = None  # None → experiment config data_dir

PATH = "pred"
TARGET = "p99_ae"  # p99_ae | mse | struct | joint | mae | peak_sharp_px
N_SAMPLES = 2000
TEST_FRAC = 0.4
BUDGET_K = 80  # equal "ECAD" budget on the holdout
SEED = 0

GP_KIND = "sklearn"  # sklearn (fast) or svgp
N_INDUCING = 128
SVGP_EPOCHS = 40
KAPPA = 0.5
SCORE_MODE = "mu"  # mu | ucb | ucb_novelty — A/B reports all three GP variants anyway
NOVELTY_WEIGHT = 0.25  # used only for ucb_novelty row in the report
NOVELTY_K = 10
JOINT_ALPHA_HM = 1.0
JOINT_ALPHA_IMP = 1.0

INCLUDE_MC = False  # True → also score holdout with MC auto_bad (slower)
MC_MAX = 120  # cap MC layouts if INCLUDE_MC (each layout is expensive)

DEVICE = "cpu"  # use "cuda:0" only if free; default CPU avoids training contention
OUT_DIR = "active_learning_pi/runs/acquisition_ab_validation_exp059"

# ── End CONFIG ────────────────────────────────────────────────────────────────


def _topk_mean(scores: np.ndarray, y: np.ndarray, k: int) -> float:
    k = max(1, min(int(k), len(scores)))
    order = np.argsort(-scores)[:k]
    return float(y[order].mean())


def _spearman(scores: np.ndarray, y: np.ndarray) -> float | None:
    if len(scores) < 3 or np.std(scores) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(spearmanr(scores, y).statistic)


def main() -> None:
    from active_learning_pi.al.gp_error_surrogate import (
        build_fit_loader,
        build_joint_target,
        collect_residuals,
        make_regressor,
    )
    from active_learning_pi.al.inference_pool import _load_engine
    from sklearn.preprocessing import StandardScaler

    groot = Path.cwd()
    device = torch.device(DEVICE if DEVICE else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = groot / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "experiment_dir": EXPERIMENT,
        "checkpoint_path": CHECKPOINT or f"{EXPERIMENT}/checkpoints/last_model.pt",
        "data_dir": DATA_DIR,
    }
    print(f"exp={EXPERIMENT} device={device} target={TARGET} path={PATH}")
    engine = _load_engine(cfg, groot, device)
    import importlib

    pkg = EXPERIMENT.replace("/", ".")
    inf = importlib.import_module(pkg + ".codes.inference_vae")
    exp_yaml = inf.load_experiment_config(Path(EXPERIMENT) / "config.yaml")
    loader = build_fit_loader(EXPERIMENT, DATA_DIR, groot, cfg_yaml=exp_yaml)

    print(f"Collecting ≤{N_SAMPLES} labeled residuals…")
    Z, Y, MHZ, KK, OCC = collect_residuals(
        engine,
        loader,
        path=PATH,
        n_samples=N_SAMPLES,
        device=device,
        cfg=exp_yaml,  # type: ignore[arg-type]
        return_occupancy=True,
    )
    if TARGET == "joint":
        Y["joint"] = build_joint_target(
            Y["mse"], Y["imp_mse"], alpha_hm=JOINT_ALPHA_HM, alpha_imp=JOINT_ALPHA_IMP
        )
    y_all = Y[TARGET]
    n = Z.shape[0]
    print(f"collected N={n} latent_dim={Z.shape[1]} y_mean={y_all.mean():.4f}")

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n)
    n_test = max(50, int(n * TEST_FRAC))
    idx_test = perm[:n_test]
    idx_fit = perm[n_test:]
    if len(idx_fit) < 80:
        raise SystemExit(f"n_fit={len(idx_fit)} too small; increase N_SAMPLES")

    # Fit GP on fit split only (manual, so we control the split)
    z_fit, y_fit = Z[idx_fit], y_all[idx_fit]
    scaler = StandardScaler().fit(z_fit)
    zs_fit = scaler.transform(z_fit)
    reg = make_regressor(
        GP_KIND, device, n_inducing=N_INDUCING, epochs=SVGP_EPOCHS, verbose=True
    ).fit(zs_fit, y_fit)

    from active_learning_pi.al.gp_error_surrogate import ErrorGPArtifact, knn_novelty

    nov_fit = knn_novelty(zs_fit, zs_fit, k=NOVELTY_K)
    artifact = ErrorGPArtifact(
        regressor=reg,
        scaler=scaler,
        path=PATH,
        target=TARGET,
        kappa=KAPPA,
        gp_kind=GP_KIND,
        n_fit=len(idx_fit),
        novelty_weight=NOVELTY_WEIGHT,
        novelty_k=NOVELTY_K,
        score_mode=SCORE_MODE,
        z_bank_scaled=zs_fit.copy(),
        novelty_mean=float(nov_fit.mean()),
        novelty_std=float(nov_fit.std() or 1.0),
    )

    z_test, y_test = Z[idx_test], y_all[idx_test]
    # Report all GP variants regardless of SCORE_MODE default
    art_mu = ErrorGPArtifact(**{**artifact.__dict__, "score_mode": "mu", "novelty_weight": 0.0})
    art_ucb = ErrorGPArtifact(**{**artifact.__dict__, "score_mode": "ucb", "novelty_weight": 0.0})
    art_nov = ErrorGPArtifact(**{**artifact.__dict__, "score_mode": "ucb_novelty"})
    _, _, _, acq_mu = art_mu.score_z(z_test)
    _, _, ucb, _ = art_ucb.score_z(z_test)
    mu, sigma, _, acq_nov = art_nov.score_z(z_test)
    random_scores = rng.rand(len(y_test))

    budget = min(BUDGET_K, len(y_test))
    methods = {
        "gp_mu_only": acq_mu,
        "gp_ucb_only": ucb,
        "gp_ucb_novelty": acq_nov,
        "random": random_scores,
    }

    report: dict = {
        "experiment": EXPERIMENT,
        "checkpoint": cfg["checkpoint_path"],
        "path": PATH,
        "target": TARGET,
        "n_total": int(n),
        "n_fit": int(len(idx_fit)),
        "n_test": int(len(idx_test)),
        "budget_k": int(budget),
        "kappa": KAPPA,
        "novelty_weight": NOVELTY_WEIGHT,
        "gp_kind": GP_KIND,
        "methods": {},
        "pass_gp_beats_random": False,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    print("\n=== Equal-budget holdout A/B ===")
    print(f"n_fit={len(idx_fit)} n_test={len(idx_test)} budget_k={budget}")
    print(f"{'method':<18}{'spearman':>10}{'top_k_mean_y':>14}{'vs_random':>12}")

    rand_top = _topk_mean(random_scores, y_test, budget)
    for name, scores in methods.items():
        sp = _spearman(scores, y_test)
        top = _topk_mean(scores, y_test, budget)
        lift = top / rand_top if rand_top > 1e-12 else None
        report["methods"][name] = {
            "spearman_vs_true_y": sp,
            "top_k_mean_true_y": top,
            "lift_vs_random_top_k": lift,
            "mean_score": float(np.mean(scores)),
        }
        sp_s = f"{sp:+.3f}" if sp is not None else "n/a"
        lift_s = f"{lift:.3f}x" if lift is not None else "n/a"
        print(f"{name:<18}{sp_s:>10}{top:>14.4f}{lift_s:>12}")

    gp_top = report["methods"]["gp_mu_only"]["top_k_mean_true_y"]
    gp_sp = report["methods"]["gp_mu_only"]["spearman_vs_true_y"]
    report["pass_gp_beats_random"] = bool(
        gp_top > rand_top and gp_sp is not None and gp_sp > 0.0
    )
    report["primary_method"] = "gp_mu_only"
    report["verdict"] = (
        "PASS — GP(mu) top-k true residual > random and Spearman>0"
        if report["pass_gp_beats_random"]
        else "FAIL — do not claim AL improvement until GP(mu) beats random on holdout"
    )
    print(f"\n=> {report['verdict']}")

    if INCLUDE_MC:
        print(f"\nOptional MC baseline on ≤{MC_MAX} holdout layouts…")
        from active_learning_pi.al.candidates import Candidate
        from active_learning_pi.al.inference_pool import predict_candidates

        n_mc = min(int(MC_MAX), len(idx_test))
        sub = idx_test[:n_mc]
        cands = []
        for j, gi in enumerate(sub):
            occ_vec = OCC[gi]
            # binary-ish occupancy for Candidate
            occ_list = [int(x > 0.5) for x in occ_vec.tolist()]
            cands.append(
                Candidate(
                    candidate_id=j,
                    occupancy=occ_list,
                    mhz=float(MHZ[gi]),
                    k=int(KK[gi]),
                )
            )
        mc_cfg = {
            "experiment_dir": EXPERIMENT,
            "checkpoint_path": cfg["checkpoint_path"],
            "acquisition_mode": "mc",
            "inference_mode": "layout",
            "pi_ref_mhz": 200.0,
            "mc_passes": 2,
            "mc_latent_passes": 2,
            "mc_decoder_passes": 4,
            "mc_decoder_dropout": True,
            "score_metric": "auto_bad",
            "calibrate_fg_max": False,
        }
        mc_rows = predict_candidates(mc_cfg, cands, groot, device=device)
        mc_scores_full = np.full(len(y_test), np.nan, dtype=np.float64)
        y_mc = y_test[:n_mc]
        mc_scores = np.asarray([float(r["badness"]) for r in mc_rows], dtype=np.float64)
        # Align: sub is first n_mc of idx_test, so positions 0..n_mc-1 in y_test
        sp = _spearman(mc_scores, y_mc)
        top = _topk_mean(mc_scores, y_mc, min(budget, n_mc))
        rand_mc = _topk_mean(rng.rand(n_mc), y_mc, min(budget, n_mc))
        lift = top / rand_mc if rand_mc > 1e-12 else None
        report["methods"]["mc_auto_bad"] = {
            "spearman_vs_true_y": sp,
            "top_k_mean_true_y": top,
            "lift_vs_random_top_k": lift,
            "n_mc": int(n_mc),
            "note": "MC compared on a subset of the holdout; budget capped to subset size",
        }
        sp_s = f"{sp:+.3f}" if sp is not None else "n/a"
        print(f"{'mc_auto_bad':<18}{sp_s:>10}{top:>14.4f}{(f'{lift:.3f}x' if lift else 'n/a'):>12}")
        # Also report GP on same subset for fair comparison
        gp_sub = acq_mu[:n_mc]
        gp_top_sub = _topk_mean(gp_sub, y_mc, min(budget, n_mc))
        report["methods"]["gp_ucb_novelty_on_mc_subset"] = {
            "top_k_mean_true_y": gp_top_sub,
            "lift_vs_mc_top_k": (gp_top_sub / top) if top > 1e-12 else None,
            "n": int(n_mc),
        }
        print(
            f"  GP vs MC on same subset: GP_top={gp_top_sub:.4f} MC_top={top:.4f} "
            f"ratio={gp_top_sub / top if top > 1e-12 else float('nan'):.3f}"
        )

    out_path = out_dir / f"ab_{TARGET}_{PATH}_{int(time.time())}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    latest = out_dir / "LATEST_acquisition_ab.json"
    latest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(f"Wrote {latest}")

    # Link into the primary AL run so DECISION_REPORT can cite equal-budget A/B.
    try:
        from active_learning_pi.al.config import load_config as _load_al_cfg
        from active_learning_pi.al.decision_report import register_acquisition_ab

        al_cfg_path = groot / "active_learning_pi/config/exp059_gp_error.json"
        if al_cfg_path.is_file():
            al_cfg = _load_al_cfg(al_cfg_path)
            dest = register_acquisition_ab(al_cfg, groot, report, src_path=latest)
            print(f"Registered A/B into AL run → {dest}")
    except Exception as exc:
        print(f"(skip AL-run A/B register: {exc})")

    sys.exit(0 if report["pass_gp_beats_random"] else 2)


if __name__ == "__main__":
    main()
