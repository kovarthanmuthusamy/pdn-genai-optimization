"""Baseline for the latent search: random placements judged by the surrogate.

What: is the search in the latent space needed, or is the surrogate with random proposals enough?
    For every target spectrum and budget K = 2..10, N_RANDOM placements are drawn at random (slots
    without repetition, a type at random for every decap), the surrogate predicts their spectra from
    the placement alone, and the N_KEEP placements with the lowest predicted worst-case ratio that the
    surrogate lets pass (ratio at most 1) become the proposals. N_RANDOM is the number of placements
    the latent search evaluates for one target and budget (a pool of 1,000 draws and 64 starting
    points over 1,200 steps), and N_KEEP its number of starting points, so both methods use the same
    surrogate effort and propose at most the same number of placements for simulation.
    The proposals are written in the format of the search (candidates.json per budget) and registered
    as run label "rand" in ch4_search_runs.json; check_search_ecad simulates and scores them with the
    same code as the search. Null: the latent search proposes placements that pass as often, reach
    as small a budget and pass as safely as this baseline.
    The baseline is drawn three times (labels rand, rand1, rand2; other random draws, nothing else
    changed), as the search is run three times. A label whose proposals exist already is kept.
Usage: python -m thesis_code.s12_evaluation.check_random_surrogate     (CONFIG-only; GPU recommended;
       then check_search_ecad)
Outputs: random_surrogate/<target>/K??/candidates.json (+ run_config.json) for "rand",
    random_surrogate_<label>/... for the repeats; their entries in ch4_search_runs.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from thesis_code.s12_evaluation import common as c

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
K_VALUES = range(2, 11)
N_RANDOM = 1000 + 64 * 1200          # placements the latent search evaluates per target and budget
N_KEEP = 64                          # its number of starting points
SEEDS = {"rand": 20261011, "rand1": 20261012, "rand2": 20261013}   # label -> seed of the random draws
STORE = "random_surrogate"           # proposals of "rand"; the repeats go to STORE_<label>
RUNS_FILE = "ch4_search_runs.json"
# =============================================================================


def out_dir(label: str) -> Path:
    return c.ev_out(STORE if label == "rand" else f"{STORE}_{label}")


def main() -> None:
    from thesis_code.s09_inference.placement_encoder import margin as measured_margin
    tg = c.targets()
    sur = c.Surrogate()
    for label, seed in SEEDS.items():
        if all((out_dir(label) / t / "run_config.json").is_file() for t in tg):
            print(f"{label}: proposals exist already, kept", flush=True)
            continue
        rng = np.random.default_rng(seed)
        t0 = time.time()
        pred = {}
        for k in K_VALUES:
            cd = np.zeros((N_RANDOM, c.N_SLOTS), dtype=np.int64)
            slots = np.argsort(rng.random((N_RANDOM, c.N_SLOTS)), axis=1)[:, :k]
            np.put_along_axis(cd, slots, rng.integers(1, 4, size=(N_RANDOM, k)), axis=1)
            cd = np.unique(cd, axis=0)
            pred[k] = (cd, sur.spectra(cd))
            print(f"{label} K={k}: {len(cd):,} distinct random placements predicted", flush=True)
        seconds = time.time() - t0
        runs_in = c.ev_in(RUNS_FILE)
        runs = json.loads(runs_in.read_text()) if runs_in.is_file() else {}
        for t, zt in tg.items():
            margin = float(measured_margin(c.TARGET_PATHS[t]))
            run = out_dir(label) / t
            for k in K_VALUES:
                cd, z = pred[k]
                rho = c.worst_ratio(z, zt)
                order = np.argsort(rho)[:N_KEEP]
                rows = [{"seed": int(i), "step": 0, "codes": cd[i].tolist(), "claimed_worst_ratio": float(rho[i]),
                         "recheck_worst_ratio": float(rho[i]), "recheck_pass": bool(rho[i] <= 1 - margin)}
                        for i in order if rho[i] <= 1.0]
                c.save_json(run / f"K{k:02d}" / "candidates.json",
                            {"K": int(k), "n_random": int(len(cd)), "n_predicted_pass": int((rho <= 1).sum()),
                             "n_predicted_pass_with_margin": int((rho <= 1 - margin).sum()), "n_distinct_layouts": len(rows), "rows": rows})
                print(f"  {label} {t} K={k}: predicted to pass {int((rho <= 1).sum()):,} of {len(cd):,}; proposals {len(rows)}", flush=True)
            c.save_json(run / "run_config.json", {"method": "random placements judged by the surrogate",
                                                  "n_random": N_RANDOM, "n_keep": N_KEEP, "seed": seed, "recheck_margin": margin,
                                                  "target_impedance_path": str(c.TARGET_PATHS[t])})
            runs[f"{label}/{t}"] = {"run_dir": str(run), "seconds": round(seconds / len(tg), 1)}
        c.save_json(c.ev_out(RUNS_FILE), runs)


if __name__ == "__main__":
    main()
