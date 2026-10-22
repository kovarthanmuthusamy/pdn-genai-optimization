"""eCADSTAR check of every placement the search proposes.

What: every run of run_search (and the baseline of check_random_surrogate) lists per target spectrum
    and budget K the proposed placements with two predicted worst-case ratios: the search's own claim
    and the one from the placement alone. EVERY proposal is simulated with eCADSTAR (spectrum at the
    IC port); a proposal is right only if the SIMULATED spectrum stays below the target at all 231
    points. Because all proposals are simulated, two ways of using the placement-alone ratio are read
    from the same simulations:
      safe    only proposals whose placement-alone ratio is at most 1 - margin (margin = the smallest
              one with no false pass on the test placements, per target);
      direct  no margin: every proposal is simulated and the simulation decides.
    Measures: smallest budget with a verified placement; how safely it passes (best simulated
    worst-case ratio per budget, against the best of the dataset's simulated placements of that
    budget); share of the proposals that pass (against the share of the dataset's random placements
    that pass). Also: error of the two predicted ratios on the proposals, decap types of the verified
    placements, run-to-run spread (labels s0, s1, s2), and placements of T1 decaps only at the
    smallest budget (the choice of the slots alone).
    Null for "the search finds better placements than chance": the share of its proposals that pass
    equals the share of random placements of the same budget that pass.
    A placement is simulated once (store search_ecad/spectra.npz), so runs that propose the same
    placement share the simulation.
Usage: python -m thesis_code.s12_evaluation.check_search_ecad     (CONFIG-only; resumable; needs the
       eCADSTAR host for proposals not in the store)
Outputs: store search_ecad/ (when new); ch4_search.json
"""
from __future__ import annotations

import json

import numpy as np

from thesis_code.s12_evaluation import common as c
from thesis_code.s12_evaluation import ecad

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
LABELS = ["s0", "s1", "s2", "rand", "nograd", "noste", "n8", "n16", "n32", "rand1", "rand2"]   # simulated in this order
RUNS_FILE = "ch4_search_runs.json"   # written by run_search and check_random_surrogate
STORE = "search_ecad"
SIMULATE = True                      # False: analyse what is simulated already, skip the rest
OUT_FILE = "ch4_search.json"
# =============================================================================


def load_runs() -> dict:
    """label -> target -> {margin, seconds, K -> rows [codes, claimed ratio, re-check ratio, starting point]}."""
    reg = json.loads(c.ev_in(RUNS_FILE).read_text())
    out: dict = {}
    for key, info in reg.items():
        label, target = key.split("/")
        run = c.search_run_dir(info["run_dir"])
        cfg = json.loads((run / "run_config.json").read_text())
        cells = {}
        for k_dir in sorted(run.glob("K*")):
            f = k_dir / "candidates.json"
            rows = json.loads(f.read_text())["rows"] if f.is_file() else []
            cells[int(k_dir.name[1:])] = rows
        tim = run / "timing_summary.json"
        out.setdefault(label, {})[target] = {"margin": float(cfg["recheck_margin"]), "seconds": info.get("seconds"),
                                              "timing": json.loads(tim.read_text()) if tim.is_file() else None, "cells": cells}
    return out


def _types(codes: np.ndarray) -> dict:
    return {"n_types": int(sum((codes == t).any() for t in (1, 2, 3))), **{f"T{t}": int((codes == t).sum()) for t in (1, 2, 3)}}


def main() -> None:
    tg = c.targets()
    runs = load_runs()
    labels = [l for l in LABELS if l in runs]
    st = ecad.store(STORE)
    for l in labels:                                   # simulate label by label, most important first
        codes = np.array([r["codes"] for t in runs[l].values() for rows in t["cells"].values() for r in rows], dtype=np.int64)
        if len(codes) == 0:
            continue
        miss = ecad.n_missing(st, codes)
        print(f"{l}: {len(codes)} proposals, {miss} not simulated yet", flush=True)
        if miss and SIMULATE:
            ecad.spectra_cached(st, codes)
    d = np.load(st / "spectra.npz")
    sim = {x.tobytes(): z for x, z in zip(d["codes"], d["values"])}

    # the dataset as the reference: share of its placements that pass, and its best placement, per K
    idx_ids = c.split_ids("train") + c.split_ids("test")
    ds_codes, ds_spec = c.codes_of(idx_ids), c.spectra_of(idx_ids)
    ds_k = (ds_codes > 0).sum(1)
    dataset = {t: {int(k): {"n": int((ds_k == k).sum()), "passing_pct": c.share(c.worst_ratio(ds_spec[ds_k == k], zt) <= 1),
                            "best_ratio": float(c.worst_ratio(ds_spec[ds_k == k], zt).min())} for k in range(2, 11)}
               for t, zt in tg.items()}

    res: dict = {"dataset": dataset, "labels": {}, "n_simulated": int(len(d["codes"]))}
    for l in labels:
        res["labels"][l] = {}
        for t, run in runs[l].items():
            zt, m = tg[t], run["margin"]
            cells, cl_err, rc_err, cl_sgn, rc_sgn = {}, [], [], [], []
            for k, rows in sorted(run["cells"].items()):
                recs = []
                for r in rows:
                    cd = np.asarray(r["codes"], dtype=np.int64)
                    if cd.tobytes() not in sim:
                        continue
                    rho = float(c.worst_ratio(sim[cd.tobytes()], zt))
                    recs.append({"claimed": float(r["claimed_worst_ratio"]), "recheck": float(r["recheck_worst_ratio"]),
                                 "sim": rho, "codes": r["codes"], **_types(cd)})
                    cl_err.append(abs(recs[-1]["claimed"] / rho - 1) * 100); cl_sgn.append((recs[-1]["claimed"] / rho - 1) * 100)
                    rc_err.append(abs(recs[-1]["recheck"] / rho - 1) * 100); rc_sgn.append((recs[-1]["recheck"] / rho - 1) * 100)
                safe = [x for x in recs if x["recheck"] <= 1 - m]
                zero = [x for x in recs if x["recheck"] <= 1]
                first = min(recs, key=lambda x: x["recheck"]) if recs else None      # the one the flow would return first
                good = [x for x in recs if x["sim"] <= 1]
                cells[int(k)] = {
                    "n_proposed": len(rows), "n_simulated": len(recs), "n_pass": len(good),
                    "pass_pct": c.share([x["sim"] <= 1 for x in recs]) if recs else None,
                    "best_ratio": min((x["sim"] for x in recs), default=None),
                    "safe": {"n": len(safe), "n_pass": sum(x["sim"] <= 1 for x in safe),
                             "best_ratio": min((x["sim"] for x in safe), default=None)},
                    "margin0": {"n": len(zero), "n_pass": sum(x["sim"] <= 1 for x in zero)},
                    "first": None if first is None else {"recheck": first["recheck"], "sim": first["sim"], "passes": first["sim"] <= 1,
                                                         "kept_safe": first["recheck"] <= 1 - m},
                    "false_pass_safe": sum(x["sim"] > 1 for x in safe),
                    "good_rejected_by_margin": sum(x["sim"] <= 1 and x["recheck"] > 1 - m for x in recs),
                    "types_pass": {"one": sum(x["n_types"] == 1 for x in good), "two": sum(x["n_types"] == 2 for x in good),
                                   "three": sum(x["n_types"] == 3 for x in good),
                                   **{f"T{i}": sum(x[f"T{i}"] for x in good) for i in (1, 2, 3)}},
                    "rows": recs}
            ks = sorted(cells)
            tot = lambda f: int(sum(f(cells[k]) for k in ks))
            res["labels"][l][t] = {
                "margin": m, "seconds": run["seconds"], "timing": run["timing"], "cells": cells,
                "smallest_budget_direct": min((k for k in ks if cells[k]["n_pass"] > 0), default=None),
                "smallest_budget_safe": min((k for k in ks if cells[k]["safe"]["n_pass"] > 0), default=None),
                "smallest_budget_first": min((k for k in ks if cells[k]["first"] and cells[k]["first"]["passes"]), default=None),
                "n_proposed": tot(lambda x: x["n_proposed"]), "n_simulated": tot(lambda x: x["n_simulated"]),
                "n_pass": tot(lambda x: x["n_pass"]), "n_safe": tot(lambda x: x["safe"]["n"]),
                "n_safe_pass": tot(lambda x: x["safe"]["n_pass"]), "n_margin0": tot(lambda x: x["margin0"]["n"]),
                "n_margin0_pass": tot(lambda x: x["margin0"]["n_pass"]),
                "claimed_ratio_error_typical_pct": c.typical(cl_err), "claimed_ratio_signed_typical_pct": c.typical(cl_sgn),
                "recheck_ratio_error_typical_pct": c.typical(rc_err), "recheck_ratio_error_9of10_pct": c.nine_of_ten(rc_err),
                "recheck_ratio_signed_typical_pct": c.typical(rc_sgn)}
            s = res["labels"][l][t]
            print(f"{l:>7} {t}: proposed {s['n_proposed']:>4}  pass {s['n_pass']:>4}  safe {s['n_safe']:>4} (pass {s['n_safe_pass']:>4})  "
                  f"smallest budget direct {s['smallest_budget_direct']} / safe {s['smallest_budget_safe']} / first {s['smallest_budget_first']}",
                  flush=True)

    # ---- the slots alone: placements of T1 decaps only, at the smallest budget that can meet a target ----
    # At these budgets the capacitance law restricts the types (0.6 ohm: three T1; 0.5 ohm: at least three
    # T1 of four; 0.8 ohm: at least one T1 of three), so the share of all proposals that pass mixes the
    # choice of the type with the choice of the slots.
    # Restricting both sides to placements of K T1 decaps leaves the slots: the proposals against random
    # placements of K T1 decaps (the dataset's; for three decaps also all simulated ones). Null: the same
    # share passes. "search" pools the three runs s0, s1, s2.
    bp_f = c.ev_in("ch4_budget_proof.json")
    if bp_f.is_file():
        bp = json.loads(bp_f.read_text())
        res["t1_only"] = {}
        for t, zt in tg.items():
            k = int(bp["targets"][t]["smallest_K_with_passing_placement"])
            q = (ds_k == k) & ((ds_codes == 1).sum(1) == k)
            blk = {"K": k, "dataset": {"n": int(q.sum()), "n_pass": int((c.worst_ratio(ds_spec[q], zt) <= 1).sum())}}
            if k == 3:
                blk["all_simulated"] = {"n": int(bp["t1_triples"]["n_simulated"]), "n_pass": int(bp["t1_triples"]["targets"][t]["n_pass"])}
            groups = {l: [l] for l in labels}
            groups["search"] = [l for l in ("s0", "s1", "s2") if l in labels]
            groups["random"] = [l for l in ("rand", "rand1", "rand2") if l in labels]
            for g, ls in groups.items():
                rows = [x for l in ls for x in res["labels"][l].get(t, {}).get("cells", {}).get(k, {}).get("rows", [])]
                t1 = [x for x in rows if x["T1"] == k]
                blk[g] = {"n_proposed": len(rows), "n_pass": sum(x["sim"] <= 1 for x in rows), "n_t1_only": len(t1),
                          "n_t1_only_pass": sum(x["sim"] <= 1 for x in t1),
                          "n_t1_only_distinct": len({tuple(x["codes"]) for x in t1})}
            res["t1_only"][t] = blk
            print(f"T1 only, {t}, K={k}: dataset {blk['dataset']['n_pass']}/{blk['dataset']['n']}"
                  + (f", all simulated {blk['all_simulated']['n_pass']}/{blk['all_simulated']['n']}" if k == 3 else "") + "; "
                  + "; ".join(f"{g} {blk[g]['n_t1_only_pass']}/{blk[g]['n_t1_only']} (of {blk[g]['n_proposed']} proposals)" for g in groups), flush=True)

    # ---- how much the three runs of the search overlap ---------------------------------------------
    per_run = {l: {tuple(x["codes"]) for t in res["labels"][l].values() for cell in t["cells"].values() for x in cell["rows"]}
               for l in ("s0", "s1", "s2") if l in labels}
    if len(per_run) > 1:
        union = set().union(*per_run.values())
        res["runs_overlap"] = {"distinct_per_run": {l: len(v) for l, v in per_run.items()}, "distinct_all_runs": len(union),
                               "in_more_than_one_run": sum(sum(x in v for v in per_run.values()) > 1 for x in union)}
        print("runs overlap:", res["runs_overlap"], flush=True)

    # ---- the three sets of random draws of the reference: overlap, and what holds in every one ------
    draws = [l for l in ("rand", "rand1", "rand2") if l in labels]
    if len(draws) > 1:
        per_draw = {l: {tuple(x["codes"]) for t in res["labels"][l].values() for cell in t["cells"].values() for x in cell["rows"]}
                    for l in draws}
        union = set().union(*per_draw.values())
        res["draws_overlap"] = {"distinct_per_draw": {l: len(v) for l, v in per_draw.items()}, "distinct_all_draws": len(union),
                                "in_more_than_one_draw": sum(sum(x in v for v in per_draw.values()) > 1 for x in union)}
        print("draws overlap:", res["draws_overlap"], flush=True)
    c.save_json(c.ev_out(OUT_FILE), res)
    print("->", c.ev_out(OUT_FILE))


if __name__ == "__main__":
    main()
