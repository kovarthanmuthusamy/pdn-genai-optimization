"""Where to add the next decap: the surrogate's ranking against eCADSTAR.

What: a designer has a placement that fails the target and may add one decap. For N_CASES_PER_TARGET
    test placements per target spectrum that fail by a small amount (worst-case ratio between RHO_MIN
    and RHO_MAX, K = 4..8, first slot empty), EVERY possible addition -- each empty slot with each of
    the three types -- is simulated. This is the full ground truth of the question, so no sampling
    enters: the surrogate ranks the same additions from the placement alone, and the two rankings are
    compared directly.
    Measures per case: rank correlation of the predicted with the simulated worst-case ratio over all
    additions; whether the surrogate's best addition (and its best five) pass in the simulation; where
    the surrogate's best addition stands in the simulated order; the loss against the truly best
    addition. References: an addition picked at random (the share of all additions that pass), and
    the designer's rule (a T1 decap in the empty slot closest to the IC port).
    Null: the ranking of the surrogate is no better than chance (rank correlation 0, its best addition
    passes as often as a random one).
Usage: python -m thesis_code.s12_evaluation.check_next_decap_ecad     (CONFIG-only; resumable; about
       13 min of eCADSTAR when the store is new)
Outputs: store next_decap_ecad/ (when new); ch4_next_decap.json (+ .npz)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from thesis_code.s12_evaluation import common as c
from thesis_code.s12_evaluation import ecad

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
N_CASES_PER_TARGET = 2
K_VALUES = range(4, 9)
RHO_MIN, RHO_MAX = 1.05, 1.40
SEED = 20261011
STORE = "next_decap_ecad"
# =============================================================================


def prepare(st: Path) -> dict:
    rng = np.random.default_rng(SEED)
    test = c.split_ids("test")
    codes, spec = c.codes_of(test), c.spectra_of(test)
    K = (codes > 0).sum(1)
    cases, rows, used_k = [], [], set()
    for t, zt in c.targets().items():
        rho = c.worst_ratio(spec, zt)
        picked = 0
        for i in rng.permutation(len(test)):
            if (K[i] in K_VALUES and codes[i][0] == 0 and RHO_MIN < rho[i] <= RHO_MAX and (t, int(K[i])) not in used_k
                    and not any(cs["design_id"] == test[i] for cs in cases)):
                used_k.add((t, int(K[i])))
                ci = len(cases)
                cases.append({"design_id": test[i], "target": t, "K": int(K[i]), "codes": codes[i].tolist(),
                              "ratio": float(rho[i])})
                for s in np.flatnonzero(codes[i] == 0):
                    for ty in (1, 2, 3):
                        cd = codes[i].copy(); cd[s] = ty
                        rows.append({"case": ci, "slot": int(s), "type": ty, "codes": cd.tolist()})
                picked += 1
                if picked == N_CASES_PER_TARGET:
                    break
    man = {"seed": SEED, "cases": cases, "spectra": ecad.assign(rows)}
    c.save_json(ecad.writable(st) / "manifest.json", man)
    print(f"{len(cases)} cases -> {len(rows)} spectra to simulate", flush=True)
    return man


def analyze(st: Path, man: dict) -> dict:
    from scipy.stats import spearmanr
    tg = c.targets()
    xy, ic = c.slot_xy_mm()
    d_ic = np.hypot(*(xy - ic).T)
    Z = ecad.read_spectra(st, man["spectra"])
    cd = np.array([r["codes"] for r in man["spectra"]])
    case = np.array([r["case"] for r in man["spectra"]]); slot = np.array([r["slot"] for r in man["spectra"]])
    typ = np.array([r["type"] for r in man["spectra"]])
    sur = c.Surrogate()
    P = sur.spectra(cd)
    train = {x.tobytes() for x in c.codes_of(c.split_ids("train"))}
    out = {"cases": []}
    for ci, cs in enumerate(man["cases"]):
        m = case == ci
        zt = tg[cs["target"]]
        rs, rp = c.worst_ratio(Z[m], zt), c.worst_ratio(P[m], zt)
        order_s, order_p = np.argsort(rs), np.argsort(rp)
        rank_s = np.empty(len(rs), int); rank_s[order_s] = np.arange(1, len(rs) + 1)
        best = order_p[0]
        sl, ty = slot[m], typ[m]
        empty = np.unique(sl)
        rule = np.flatnonzero((sl == empty[np.argmin(d_ic[empty])]) & (ty == 1))[0]
        base_pred = float(c.worst_ratio(sur.spectra(np.array([cs["codes"]])), zt)[0])
        out["cases"].append({
            **{k: cs[k] for k in ("design_id", "target", "K", "ratio")}, "ratio_predicted": base_pred,
            "n_additions": int(m.sum()), "n_additions_in_training_set": int(sum(x.tobytes() in train for x in cd[m])),
            "additions_passing_pct": c.share(rs <= 1), "n_additions_passing": int((rs <= 1).sum()),
            "rank_correlation": float(spearmanr(rs, rp)[0]),
            "ratio_error_typical_pct": c.typical(np.abs(rp / rs - 1) * 100),
            "best_simulated": {"slot": int(sl[order_s[0]]) + 1, "type": int(ty[order_s[0]]), "ratio": float(rs[order_s[0]])},
            "surrogate_best": {"slot": int(sl[best]) + 1, "type": int(ty[best]), "ratio_predicted": float(rp[best]),
                               "ratio_simulated": float(rs[best]), "passes": bool(rs[best] <= 1), "rank_simulated": int(rank_s[best]),
                               "loss_vs_best_pct": float((rs[best] / rs[order_s[0]] - 1) * 100)},
            "surrogate_top5_passing": int((rs[order_p[:5]] <= 1).sum()),
            "surrogate_top5_in_simulated_top5": int(len(set(order_p[:5]) & set(order_s[:5]))),
            "surrogate_says_pass": int((rp <= 1).sum()), "of_these_pass": int(((rp <= 1) & (rs <= 1)).sum()),
            "rule_of_thumb": {"slot": int(sl[rule]) + 1, "ratio_simulated": float(rs[rule]), "passes": bool(rs[rule] <= 1),
                              "rank_simulated": int(rank_s[rule])},
            "by_type_passing_pct": {int(t): c.share(rs[ty == t] <= 1) for t in (1, 2, 3)}})
    cc = out["cases"]
    out["summary"] = {"n_cases": len(cc), "rank_correlation_typical": c.typical([x["rank_correlation"] for x in cc]),
                      "rank_correlation_min": float(min(x["rank_correlation"] for x in cc)),
                      "surrogate_best_passes": int(sum(x["surrogate_best"]["passes"] for x in cc)),
                      "rule_passes": int(sum(x["rule_of_thumb"]["passes"] for x in cc)),
                      "random_addition_passes_pct": c.typical([x["additions_passing_pct"] for x in cc]),
                      "surrogate_best_rank_typical": c.typical([x["surrogate_best"]["rank_simulated"] for x in cc]),
                      "loss_vs_best_typical_pct": c.typical([x["surrogate_best"]["loss_vs_best_pct"] for x in cc])}
    c.save_json(c.ev_out("ch4_next_decap.json"), out)
    np.savez_compressed(c.ev_out("ch4_next_decap.npz"), Z=Z, P=P, codes=cd, case=case, slot=slot, type=typ)
    print(json.dumps(out, indent=1))
    return out


def main() -> None:
    st = ecad.store(STORE)
    man_p = st / "manifest.json"
    man = json.loads(man_p.read_text()) if man_p.is_file() else prepare(st)
    ecad.simulate(st, man["spectra"], [])
    analyze(st, man)


if __name__ == "__main__":
    main()
