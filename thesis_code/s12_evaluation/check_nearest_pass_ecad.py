"""The nearest placement that passes, checked with eCADSTAR.

What: a designer proposes a placement. If it fails the target, the assistant (s11_assistant.suggest:
    find_fixes) returns the nearest placements that the surrogate lets pass: fewest changed decaps
    first, a new type in place before a move, the number of decaps unchanged; all placements one
    change away are tried, then two changes, then a sample of three.
    Cases: N_CASES test placements per target spectrum that fail it in simulation by up to RHO_MAX
    (K = K_MIN..10, first slot empty). For every case the first N_SHOW proposals are simulated, in
    both ways of using the surrogate as the judge:
      safe    with the margin measured for the target (no false pass on the test placements);
      direct  without a margin.
    Measures: how often a proposal within three changes exists; how many decaps it changes and how
    often a type change alone is enough; how often the proposals pass in the simulation.
    Is the proposal really the NEAREST one? For N_TRUTH cases every placement one change away is
    simulated (all type changes and all moves, a few hundred per case), which is the complete answer
    for that distance: the number of one-change placements that pass, and whether the assistant found
    one of them when there was one. Null: the assistant's proposals pass no more often than a random
    placement one change away.
    Also the first step of the assistant, the check of the proposed placement itself: the verdict of
    the surrogate for the cases (all of which fail in simulation).
Usage: python -m thesis_code.s12_evaluation.check_nearest_pass_ecad     (CONFIG-only; GPU recommended for
       the proposals, then about 25 min of eCADSTAR when the store is new; resumable)
Outputs: store nearest_pass_ecad/ (when new); ch4_nearest_pass.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from thesis_code.s12_evaluation import common as c
from thesis_code.s12_evaluation import ecad

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
N_CASES = 20                 # per target spectrum
N_SHOW = 3                   # proposals simulated per case and mode
N_TRUTH = 4                  # cases with every one-change placement simulated
K_MIN = 4
RHO_MAX = 1.5
SEED = 20261011
STORE = "nearest_pass_ecad"
# =============================================================================


def one_change(codes: np.ndarray) -> np.ndarray:
    """Every placement one change away: a new type in place, or a move to an empty slot."""
    out = []
    occ, empty = np.flatnonzero(codes), np.flatnonzero(codes == 0)
    for s in occ:
        for t in (1, 2, 3):
            if t != codes[s]:
                x = codes.copy(); x[s] = t; out.append(x)
    for s in occ:
        for d in empty:
            x = codes.copy(); x[d] = x[s]; x[s] = 0; out.append(x)
    return np.array(out, dtype=np.int64)


def propose(st: Path) -> dict:
    from thesis_code.s09_inference.placement_encoder import margin as measured_margin
    from thesis_code.s11_assistant import suggest as sl
    rng = np.random.default_rng(SEED)
    f, tg = c.freqs_mhz(), c.targets()
    test = c.split_ids("test")
    codes, spec = c.codes_of(test), c.spectra_of(test)
    K = (codes > 0).sum(1)
    band = np.arange(len(f))
    cases = []
    for t, zt in tg.items():
        rho = c.worst_ratio(spec, zt)
        pool = np.flatnonzero((K >= K_MIN) & (codes[:, 0] == 0) & (rho > 1) & (rho <= RHO_MAX))
        pick = rng.choice(pool, size=min(N_CASES, len(pool)), replace=False)
        margin = float(measured_margin(c.TARGET_PATHS[t]))
        judges = {mode: sl.load_judge(zt, band, margin=m, target_path=c.TARGET_PATHS[t])
                  for mode, m in (("safe", margin), ("direct", 0.0))}
        for i in pick:
            case = {"design_id": test[i], "target": t, "K": int(K[i]), "codes": codes[i].tolist(), "ratio": float(rho[i]),
                    "margin": margin, "modes": {}}
            for mode, judge in judges.items():
                t0 = time.time()
                ok, _, ratios = judge.assess(codes[i][None])
                # a placement the judge lets pass is not repaired (here a false pass: every case fails in simulation)
                found = [] if ok[0] else sl.find_fixes(codes[i], judge, 3, max_radius=3, min_suggestions=1, seed=SEED, log=None)
                case["modes"][mode] = {
                    "verdict_pass": bool(ok[0]), "ratio_predicted": float(next(iter(ratios.values()))[0]),
                    "n_found": len(found), "seconds": round(time.time() - t0, 2),
                    "proposals": [{"codes": x["codes"].tolist(), "changed": int(x["edits"]), "moved": int(x["moves"]),
                                   "ratio_predicted": float(next(iter(x["ratios"].values()))),
                                   "summary": sl.edit_summary(codes[i], x["codes"])} for x in found[:N_SHOW]]}
            cases.append(case)
            print(f"  {t} K={case['K']} rho={case['ratio']:.2f}: " + "; ".join(
                f"{m}: {v['n_found']} found" + (f", first changes {v['proposals'][0]['changed']} ({v['proposals'][0]['moved']} moved)" if v["proposals"] else "")
                for m, v in case["modes"].items()), flush=True)
    per_t = [[n for n, cs in enumerate(cases) if cs["target"] == t] for t in tg]          # truth cases: spread over the targets
    truth = [lst[r] for r in range(N_TRUTH) for lst in per_t if r < len(lst)][:N_TRUTH]
    man = {"seed": SEED, "cases": cases, "truth_cases": truth}
    c.save_json(ecad.writable(st) / "manifest.json", man)
    return man


def analyze(st: Path, man: dict) -> dict:
    tg = c.targets()
    cases = man["cases"]
    prop = np.array([p["codes"] for cs in cases for m in cs["modes"].values() for p in m["proposals"]], dtype=np.int64).reshape(-1, c.N_SLOTS)
    near = {j: one_change(np.array(cases[j]["codes"])) for j in man["truth_cases"]}
    allc = np.concatenate([prop] + list(near.values()))
    Z = ecad.spectra_cached(st, allc)
    sim = {x.tobytes(): z for x, z in zip(allc, Z)}
    sur = c.Surrogate()
    out: dict = {"n_cases": len(cases), "modes": {}, "truth": []}
    for mode in ("safe", "direct"):
        res = {}
        for t in list(tg) + ["all"]:
            cs = [x for x in cases if t in ("all", x["target"])]
            first = [x["modes"][mode]["proposals"][0] for x in cs if x["modes"][mode]["proposals"]]
            allp = [p for x in cs for p in x["modes"][mode]["proposals"]]
            rho = lambda p, x: float(c.worst_ratio(sim[np.array(p["codes"], dtype=np.int64).tobytes()], tg[x["target"]]))
            first_pass = [rho(x["modes"][mode]["proposals"][0], x) <= 1 for x in cs if x["modes"][mode]["proposals"]]
            all_pass = [rho(p, x) <= 1 for x in cs for p in x["modes"][mode]["proposals"]]
            any_pass = [any(rho(p, x) <= 1 for p in x["modes"][mode]["proposals"]) for x in cs if x["modes"][mode]["proposals"]]
            res[t] = {"n_cases": len(cs), "verdict_false_pass": int(sum(x["modes"][mode]["verdict_pass"] for x in cs)),
                      "with_proposal_pct": c.share([bool(x["modes"][mode]["proposals"]) for x in cs if not x["modes"][mode]["verdict_pass"]]),
                      "first_changed": {int(k): int(sum(p["changed"] == k for p in first)) for k in (1, 2, 3)},
                      "first_type_only": int(sum(p["moved"] == 0 for p in first)),
                      "first_passes_pct": c.share(first_pass), "n_first": len(first),
                      "proposals_pass_pct": c.share(all_pass), "n_proposals": len(allp),
                      "case_has_passing_proposal_pct": c.share(any_pass),
                      "seconds_typical": c.typical([x["modes"][mode]["seconds"] for x in cs])}
        out["modes"][mode] = res
    for j in man["truth_cases"]:
        cs = cases[j]
        zt = tg[cs["target"]]
        nb = near[j]
        rs = c.worst_ratio(np.stack([sim[x.tobytes()] for x in nb]), zt)
        rp = c.worst_ratio(sur.spectra(nb), zt)
        moved = ((nb > 0) != (np.array(cs["codes"]) > 0)).any(1)
        rec = {"design_id": cs["design_id"], "target": cs["target"], "K": cs["K"], "ratio": cs["ratio"],
               "n_one_change": int(len(nb)), "n_pass": int((rs <= 1).sum()), "n_pass_type_only": int(((rs <= 1) & ~moved).sum()),
               "random_one_change_passes_pct": c.share(rs <= 1), "best_ratio": float(rs.min()),
               "ratio_error_typical_pct": c.typical(np.abs(rp / rs - 1) * 100)}
        for mode, m in (("safe", cs["margin"]), ("direct", 0.0)):
            acc = rp <= 1 - m
            first = cs["modes"][mode]["proposals"][0] if cs["modes"][mode]["proposals"] else None
            rec[mode] = {"accepted": int(acc.sum()), "accepted_pass": int((acc & (rs <= 1)).sum()),
                         "true_passes_accepted_pct": c.share(acc[rs <= 1]) if (rs <= 1).any() else None,
                         "first_proposal_changed": None if first is None else first["changed"],
                         "found_one_change_fix": bool(first is not None and first["changed"] == 1),
                         "missed_a_one_change_fix": bool((rs <= 1).any() and (first is None or first["changed"] > 1))}
        out["truth"].append(rec)
    c.save_json(c.ev_out("ch4_nearest_pass.json"), {**out, "cases": cases})
    print(json.dumps(out, indent=1))
    return out


def main() -> None:
    st = ecad.store(STORE)
    man_p = st / "manifest.json"
    man = json.loads(man_p.read_text()) if man_p.is_file() else propose(st)
    analyze(st, man)


if __name__ == "__main__":
    main()
