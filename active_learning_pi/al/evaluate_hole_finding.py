"""Solid hole-finding evaluation for residual-GP active learning.

Question answered
-----------------
Does acquisition preferentially select layouts where the VAE is *actually wrong*
("holes"), better than chance / random at equal budget?

This is stronger than within-selected Spearman on an ECAD batch (which only ranks
among already-selected layouts). Here every holdout layout is labeled, so we can
compare GP's top-k against the *true* worst-k oracle and against random.

Primary metrics (PASS criteria)
-------------------------------
1. **Enrichment / hole capture** at budget K:
   Among the true top-Q% worst residuals on the holdout, what fraction fall in
   GP's top-K? Compare to random expectation K/N.
   PASS: capture_rate ≥ 1.5 × random_rate  AND  capture_rate ≥ 0.25 (at Q=20%).

2. **Equal-budget lift**:
   mean(true_y | GP top-K) / mean(true_y | random top-K)
   PASS: lift ≥ 1.3  AND  Spearman(score, y) > 0.

3. **Pool enrichment**:
   mean(true_y | GP top-K) / mean(true_y | full holdout)
   PASS: ≥ 1.5 (selected set is substantially worse than average).

4. **Structured mid-band hole** (optional honesty check):
   Hold out mid-band MHz (e.g. 150–280) from GP fit; score held-out band.
   PASS: mean(y | GP top-K ∩ midband) / mean(y | random top-K ∩ midband) ≥ 1.2
   when enough mid-band points exist.

Run:
    python active_learning_pi/al/evaluate_hole_finding.py

Writes:
    ``OUT_DIR/LATEST_hole_finding.json``
    ``OUT_DIR/HOLE_FINDING_REPORT.md``
"""
from __future__ import annotations

from repo_paths import setup_path

setup_path()

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

EXPERIMENT = "experiments/exp059_capacity_freq"
CHECKPOINT = None
DATA_DIR = None

PATH = "pred"
TARGET = "p99_ae"
N_SAMPLES = 2500
TEST_FRAC = 0.4
BUDGETS = [40, 80, 160]  # equal-budget K values to report
HOLE_QUANTILE = 0.20  # true top 20% = "holes"
SEED = 0

GP_KIND = "sklearn"
N_INDUCING = 128
SVGP_EPOCHS = 40
DEVICE = "cuda:0"  # fall back to cpu if unavailable

# Structured mid-band hole check (fit outside, score inside)
MIDBAND_MHZ = (150.0, 280.0)
RUN_STRUCTURED_MIDBAND = True

OUT_DIR = "active_learning_pi/runs/hole_finding_exp059"

# Pass thresholds
MIN_LIFT_VS_RANDOM = 1.3
MIN_POOL_ENRICHMENT = 1.5
MIN_CAPTURE_VS_RANDOM = 1.5  # relative to chance
MIN_CAPTURE_ABS = 0.25
MIN_SPEARMAN = 0.0
MIN_MIDBAND_LIFT = 1.2

# ── End CONFIG ────────────────────────────────────────────────────────────────


def _spearman(scores: np.ndarray, y: np.ndarray) -> float | None:
    if len(scores) < 5 or float(np.std(scores)) < 1e-12 or float(np.std(y)) < 1e-12:
        return None
    return float(spearmanr(scores, y).statistic)


def _topk_idx(scores: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(int(k), len(scores)))
    return np.argsort(-scores)[:k]


def _hole_mask(y: np.ndarray, q: float) -> np.ndarray:
    """True for layouts in the worst ``q`` fraction of residual y."""
    thr = float(np.quantile(y, 1.0 - q))
    return y >= thr


def _metrics_at_budget(
    scores: np.ndarray,
    y: np.ndarray,
    *,
    k: int,
    hole_q: float,
    rng: np.random.RandomState,
    n_random_trials: int = 50,
) -> dict:
    n = len(y)
    k = max(1, min(k, n))
    holes = _hole_mask(y, hole_q)
    n_holes = int(holes.sum())
    gp_idx = _topk_idx(scores, k)
    gp_capture = float(holes[gp_idx].mean()) if n_holes else 0.0
    # Chance: if we pick k at random, expected fraction of picks that are holes
    # = n_holes/n. Capture rate among GP's k = (# holes in top-k)/k.
    chance = float(n_holes) / float(n) if n else 0.0

    rand_caps = []
    rand_means = []
    for _ in range(n_random_trials):
        ridx = rng.choice(n, size=k, replace=False)
        rand_caps.append(float(holes[ridx].mean()) if n_holes else 0.0)
        rand_means.append(float(y[ridx].mean()))
    rand_cap_mean = float(np.mean(rand_caps))
    rand_y_mean = float(np.mean(rand_means))

    gp_y = float(y[gp_idx].mean())
    pool_y = float(y.mean())
    oracle_idx = np.argsort(-y)[:k]
    oracle_y = float(y[oracle_idx].mean())

    return {
        "budget_k": int(k),
        "n_holdout": int(n),
        "n_true_holes": int(n_holes),
        "hole_quantile": float(hole_q),
        "gp_top_k_mean_y": gp_y,
        "random_top_k_mean_y": rand_y_mean,
        "oracle_top_k_mean_y": oracle_y,
        "pool_mean_y": pool_y,
        "lift_vs_random": (gp_y / rand_y_mean) if rand_y_mean > 1e-12 else None,
        "pool_enrichment": (gp_y / pool_y) if pool_y > 1e-12 else None,
        "oracle_efficiency": (gp_y / oracle_y) if oracle_y > 1e-12 else None,
        "hole_capture_rate": gp_capture,
        "hole_capture_chance": chance,
        "hole_capture_random_empirical": rand_cap_mean,
        "hole_capture_lift_vs_chance": (gp_capture / chance) if chance > 1e-12 else None,
        "spearman_score_vs_y": _spearman(scores, y),
        "precision_at_k_vs_holes": gp_capture,  # same as capture among selected
        "recall_at_k_of_holes": (
            float(holes[gp_idx].sum()) / float(n_holes) if n_holes else None
        ),
    }


def _verdict(budgets: dict[str, dict], midband: dict | None) -> dict:
    # Use primary budget = median of requested keys if present, else first
    keys = sorted(budgets.keys(), key=lambda s: int(s.split("_")[-1]))
    primary_key = keys[len(keys) // 2] if keys else None
    primary = budgets.get(primary_key or "", {})

    lift = primary.get("lift_vs_random")
    enrich = primary.get("pool_enrichment")
    cap_lift = primary.get("hole_capture_lift_vs_chance")
    cap = primary.get("hole_capture_rate")
    sp = primary.get("spearman_score_vs_y")

    checks = {
        "lift_vs_random": bool(lift is not None and float(lift) >= MIN_LIFT_VS_RANDOM),
        "pool_enrichment": bool(enrich is not None and float(enrich) >= MIN_POOL_ENRICHMENT),
        "hole_capture": bool(
            cap is not None
            and cap_lift is not None
            and float(cap) >= MIN_CAPTURE_ABS
            and float(cap_lift) >= MIN_CAPTURE_VS_RANDOM
        ),
        "spearman_positive": bool(sp is not None and float(sp) > MIN_SPEARMAN),
    }
    if midband and midband.get("available"):
        mb_lift = midband.get("lift_vs_random")
        checks["midband_structured"] = bool(
            mb_lift is not None and float(mb_lift) >= MIN_MIDBAND_LIFT
        )
    else:
        checks["midband_structured"] = None  # not run / underpowered

    blocking = ["lift_vs_random", "pool_enrichment", "hole_capture", "spearman_positive"]
    fails = [c for c in blocking if checks.get(c) is False]
    if fails:
        overall = "FAIL"
        summary = (
            "Do not claim AL finds holes until these pass: " + ", ".join(fails)
        )
    elif any(checks.get(c) is None for c in blocking):
        overall = "UNCERTAIN"
        summary = "Incomplete hole-finding evidence."
    else:
        overall = "PASS"
        summary = (
            f"GP acquisition enriches true holes at equal budget "
            f"(primary {primary_key}: lift_vs_random="
            f"{lift:.2f}x, pool_enrichment={enrich:.2f}x, "
            f"hole_capture={cap:.2f}, capture_lift={cap_lift:.2f}x)."
        )
        if checks.get("midband_structured") is False:
            summary += " Mid-band structured check FAILED — scope claims off mid-band."
            overall = "PASS"  # keep overall PASS but note scoped failure
        elif checks.get("midband_structured") is True:
            summary += " Mid-band structured check also PASS."

    return {
        "overall": overall,
        "summary": summary,
        "primary_budget": primary_key,
        "checks": checks,
        "thresholds": {
            "min_lift_vs_random": MIN_LIFT_VS_RANDOM,
            "min_pool_enrichment": MIN_POOL_ENRICHMENT,
            "min_capture_vs_random": MIN_CAPTURE_VS_RANDOM,
            "min_capture_abs": MIN_CAPTURE_ABS,
            "min_spearman": MIN_SPEARMAN,
            "min_midband_lift": MIN_MIDBAND_LIFT,
        },
    }


def _render_md(report: dict) -> str:
    v = report["verdict"]
    lines = [
        "# Hole-finding evaluation report",
        "",
        f"**Generated:** {report.get('timestamp')}  ",
        f"**Experiment:** `{report.get('experiment')}`  ",
        f"**Path / target:** `{report.get('path')}` / `{report.get('target')}`  ",
        f"**Overall:** **{v.get('overall')}**  ",
        "",
        v.get("summary", ""),
        "",
        "---",
        "",
        "## What this proves",
        "",
        "On a **fully labeled holdout** (no new ECAD), acquisition scores every layout.",
        "We then ask: does GP's top-K contain **true high-error holes** better than random?",
        "",
        "| Metric | Meaning |",
        "| --- | --- |",
        "| Lift vs random | mean true error in GP top-K / random top-K |",
        "| Pool enrichment | mean true error in GP top-K / holdout mean |",
        "| Hole capture | fraction of GP top-K that are true top-Q% worst |",
        "| Capture lift | capture / chance rate |",
        "| Oracle efficiency | GP top-K mean / perfect top-K mean (≤1) |",
        "",
        "---",
        "",
        "## Pass checklist",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    for name, ok in (v.get("checks") or {}).items():
        if ok is None:
            status = "SKIP"
        else:
            status = "PASS" if ok else "FAIL"
        lines.append(f"| `{name}` | {status} |")

    lines.extend(["", "---", "", "## Results by budget", ""])
    for bkey, m in sorted(report.get("budgets", {}).items(), key=lambda kv: int(kv[0].split("_")[-1])):
        lines.extend([
            f"### Budget K={m['budget_k']}",
            "",
            f"- Spearman(score, y): **{_fmt(m.get('spearman_score_vs_y'))}**",
            f"- GP top-K mean y: **{_fmt(m.get('gp_top_k_mean_y'))}** "
            f"(random {_fmt(m.get('random_top_k_mean_y'))}, "
            f"oracle {_fmt(m.get('oracle_top_k_mean_y'))}, "
            f"pool {_fmt(m.get('pool_mean_y'))})",
            f"- Lift vs random: **{_fmt(m.get('lift_vs_random'), 3)}×**",
            f"- Pool enrichment: **{_fmt(m.get('pool_enrichment'), 3)}×**",
            f"- Hole capture (Q={m.get('hole_quantile')}): "
            f"**{_fmt(m.get('hole_capture_rate'), 3)}** "
            f"(chance {_fmt(m.get('hole_capture_chance'), 3)}, "
            f"lift {_fmt(m.get('hole_capture_lift_vs_chance'), 3)}×)",
            f"- Recall of holes: **{_fmt(m.get('recall_at_k_of_holes'), 3)}**",
            f"- Oracle efficiency: **{_fmt(m.get('oracle_efficiency'), 3)}**",
            "",
        ])

    mb = report.get("midband_structured") or {}
    lines.extend(["---", "", "## Structured mid-band hole check", ""])
    if not mb.get("available"):
        lines.append(f"_Skipped / underpowered:_ {mb.get('reason', 'n/a')}")
    else:
        lines.extend([
            f"Fit GP **outside** MHz∈[{MIDBAND_MHZ[0]}, {MIDBAND_MHZ[1]}], "
            f"score **inside** (n_fit={mb.get('n_fit')}, n_test={mb.get('n_test')}).",
            "",
            f"- Lift vs random (K={mb.get('budget_k')}): **{_fmt(mb.get('lift_vs_random'), 3)}×**",
            f"- Pool enrichment: **{_fmt(mb.get('pool_enrichment'), 3)}×**",
            f"- Hole capture lift: **{_fmt(mb.get('hole_capture_lift_vs_chance'), 3)}×**",
            f"- Spearman: **{_fmt(mb.get('spearman_score_vs_y'))}**",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## How to cite",
        "",
        "Claim only if overall is **PASS**: residual-GP acquisition on the `pred` path "
        f"with target `{report.get('target')}` enriches true VAE holes vs random at equal "
        "budget on a labeled holdout. Scope mid-band separately if structured check fails.",
        "",
        f"Artifact: `{report.get('out_json', '')}`",
        "",
    ])
    return "\n".join(lines)


def _fmt(v, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if v != v:
            return "—"
        return f"{v:.{digits}f}"
    return str(v)


def main() -> int:
    from active_learning_pi.al.gp_error_surrogate import (
        ErrorGPArtifact,
        build_fit_loader,
        collect_residuals,
        knn_novelty,
        make_regressor,
    )
    from active_learning_pi.al.inference_pool import _load_engine
    from sklearn.preprocessing import StandardScaler
    import importlib

    groot = Path.cwd()
    device = torch.device(DEVICE if torch.cuda.is_available() and "cuda" in DEVICE else "cpu")
    out_dir = groot / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "experiment_dir": EXPERIMENT,
        "checkpoint_path": CHECKPOINT or f"{EXPERIMENT}/checkpoints/last_model.pt",
        "data_dir": DATA_DIR,
    }
    print(f"exp={EXPERIMENT} device={device} target={TARGET} path={PATH}")
    engine = _load_engine(cfg, groot, device)
    pkg = EXPERIMENT.replace("/", ".")
    inf = importlib.import_module(pkg + ".codes.inference_vae")
    exp_yaml = inf.load_experiment_config(Path(EXPERIMENT) / "config.yaml")
    loader = build_fit_loader(EXPERIMENT, DATA_DIR, groot, cfg_yaml=exp_yaml)

    print(f"Collecting ≤{N_SAMPLES} labeled residuals…")
    Z, Y, MHZ, KK = collect_residuals(
        engine, loader, path=PATH, n_samples=N_SAMPLES, device=device, cfg=exp_yaml
    )
    y_all = Y[TARGET]
    finite = np.isfinite(y_all)
    Z, y_all, MHZ, KK = Z[finite], y_all[finite], MHZ[finite], KK[finite]
    n = Z.shape[0]
    print(f"collected N={n} y_mean={y_all.mean():.4f} y_std={y_all.std():.4f}")

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n)
    n_test = max(80, int(n * TEST_FRAC))
    idx_test, idx_fit = perm[:n_test], perm[n_test:]
    if len(idx_fit) < 100:
        raise SystemExit(f"n_fit={len(idx_fit)} too small")

    z_fit, y_fit = Z[idx_fit], y_all[idx_fit]
    scaler = StandardScaler().fit(z_fit)
    zs_fit = scaler.transform(z_fit)
    reg = make_regressor(
        GP_KIND, device, n_inducing=N_INDUCING, epochs=SVGP_EPOCHS, verbose=True
    ).fit(zs_fit, y_fit)
    nov = knn_novelty(zs_fit, zs_fit, k=10)
    artifact = ErrorGPArtifact(
        regressor=reg,
        scaler=scaler,
        path=PATH,
        target=TARGET,
        kappa=0.5,
        gp_kind=GP_KIND,
        n_fit=len(idx_fit),
        novelty_weight=0.0,
        novelty_k=10,
        score_mode="mu",
        z_bank_scaled=zs_fit.copy(),
        novelty_mean=float(nov.mean()),
        novelty_std=float(nov.std() or 1.0),
    )

    z_te, y_te, mhz_te = Z[idx_test], y_all[idx_test], MHZ[idx_test]
    _, _, _, acq = artifact.score_z(z_te)

    budgets = {}
    print("\n=== Hole-finding (random-split holdout) ===")
    for k in BUDGETS:
        m = _metrics_at_budget(
            acq, y_te, k=k, hole_q=HOLE_QUANTILE, rng=rng
        )
        budgets[f"k_{k}"] = m
        print(
            f"K={k:>4}  spearman={_fmt(m['spearman_score_vs_y'],3)}  "
            f"lift_vs_rand={_fmt(m['lift_vs_random'],3)}x  "
            f"pool_enr={_fmt(m['pool_enrichment'],3)}x  "
            f"capture={_fmt(m['hole_capture_rate'],3)}  "
            f"cap_lift={_fmt(m['hole_capture_lift_vs_chance'],3)}x  "
            f"oracle_eff={_fmt(m['oracle_efficiency'],3)}"
        )

    midband: dict = {"available": False, "reason": "disabled"}
    if RUN_STRUCTURED_MIDBAND:
        lo, hi = MIDBAND_MHZ
        in_mid = (MHZ >= lo) & (MHZ <= hi)
        idx_fit_mb = np.where(~in_mid)[0]
        idx_te_mb = np.where(in_mid)[0]
        if len(idx_fit_mb) < 100 or len(idx_te_mb) < 80:
            midband = {
                "available": False,
                "reason": f"underpowered n_fit={len(idx_fit_mb)} n_test={len(idx_te_mb)}",
            }
        else:
            print(
                f"\n=== Structured mid-band hole "
                f"(fit outside [{lo},{hi}] MHz, test inside) ==="
            )
            zf, yf = Z[idx_fit_mb], y_all[idx_fit_mb]
            sc = StandardScaler().fit(zf)
            reg2 = make_regressor(
                GP_KIND, device, n_inducing=N_INDUCING, epochs=SVGP_EPOCHS, verbose=False
            ).fit(sc.transform(zf), yf)
            art2 = ErrorGPArtifact(
                regressor=reg2,
                scaler=sc,
                path=PATH,
                target=TARGET,
                kappa=0.5,
                gp_kind=GP_KIND,
                n_fit=len(idx_fit_mb),
                score_mode="mu",
            )
            zt, yt = Z[idx_te_mb], y_all[idx_te_mb]
            _, _, _, acq2 = art2.score_z(zt)
            k_mb = min(80, max(20, len(yt) // 5))
            midband = _metrics_at_budget(
                acq2, yt, k=k_mb, hole_q=HOLE_QUANTILE, rng=rng
            )
            midband["available"] = True
            midband["n_fit"] = int(len(idx_fit_mb))
            midband["n_test"] = int(len(idx_te_mb))
            midband["mhz_range"] = [lo, hi]
            print(
                f"midband K={k_mb} lift={_fmt(midband['lift_vs_random'],3)}x "
                f"enrich={_fmt(midband['pool_enrichment'],3)}x "
                f"cap_lift={_fmt(midband['hole_capture_lift_vs_chance'],3)}x "
                f"spearman={_fmt(midband['spearman_score_vs_y'],3)}"
            )

    verdict = _verdict(budgets, midband if midband.get("available") else None)
    print(f"\n=> {verdict['overall']}: {verdict['summary']}")

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "experiment": EXPERIMENT,
        "checkpoint": cfg["checkpoint_path"],
        "path": PATH,
        "target": TARGET,
        "n_total": int(n),
        "n_fit": int(len(idx_fit)),
        "n_test": int(len(idx_test)),
        "hole_quantile": HOLE_QUANTILE,
        "budgets": budgets,
        "midband_structured": midband,
        "verdict": verdict,
    }
    out_json = out_dir / "LATEST_hole_finding.json"
    stamp = out_dir / f"hole_finding_{TARGET}_{PATH}_{int(time.time())}.json"
    report["out_json"] = str(out_json)
    payload = json.dumps(report, indent=2)
    out_json.write_text(payload, encoding="utf-8")
    stamp.write_text(payload, encoding="utf-8")
    md = _render_md(report)
    (out_dir / "HOLE_FINDING_REPORT.md").write_text(md, encoding="utf-8")

    # Register into AL run for decision ledger
    try:
        from active_learning_pi.al.config import load_config
        from active_learning_pi.al.paths import run_dir

        al_cfg = load_config(groot / "active_learning_pi/config/exp059_gp_error.json")
        dest = run_dir(al_cfg, groot) / "LATEST_hole_finding.json"
        dest.write_text(payload, encoding="utf-8")
        print(f"Registered → {dest}")
    except Exception as exc:
        print(f"(skip AL-run register: {exc})")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_dir / 'HOLE_FINDING_REPORT.md'}")
    return 0 if verdict["overall"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
