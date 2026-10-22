"""Four details of the search results for the three target spectra, measured on the saved results.

What:
    decoded_larger   which of the two predictions sets the search's predicted ratio: the share of
                     proposals whose predicted ratio (the larger of the two in the guided search) is
                     above the ratio from the placement alone, i.e. comes from the decoded spectrum;
                     per run and target.
    safest_vs_dataset  at 7-10 decaps, the lowest simulated ratio of the first run's proposals against
                     the best of the dataset placements of that budget. Null: the search is never safer
                     than this best.
    three_decaps_0.6 the capacitance law at its median rules out three decaps other than T1 for the
                     0.6 ohm target (247 nF against 253 nF), but its spread does not. The simulated
                     placements of three decaps with 247 nF test it directly: do any stay below
                     0.6 ohm at 1 MHz?
    low_peak_height  how high the surrogate predicts the 5-40 MHz peaks that matter which it finds
                     (test placements): predicted / simulated height.
Usage: python -m thesis_code.s12_evaluation.check_search_details     (CONFIG-only; CPU, under a minute;
       after check_search_ecad, check_budget_proof_ecad and check_accuracy)
Outputs: ch4_search_details.json
"""
from __future__ import annotations

import json

import numpy as np

from thesis_code.s12_evaluation import common as c

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
RUNS = ("s0", "s1", "s2")           # the three runs of the search (ch4_search.json labels)
LARGE_K = (7, 8, 9, 10)
OUT = "ch4_search_details.json"
# =============================================================================


def main() -> None:
    S = json.loads(c.ev_in("ch4_search.json").read_text())
    tg, f = c.targets(), c.freqs_mhz()
    out: dict = {"decoded_larger_pct": {}, "safest_vs_dataset": {}}
    for l in RUNS:
        for t in tg:
            rows = [r for cell in S["labels"][l][t]["cells"].values() for r in cell["rows"]]
            out["decoded_larger_pct"][f"{l}/{t}"] = round(100 * float(np.mean([r["claimed"] > r["recheck"] + 1e-9 for r in rows])), 2)
    for t in tg:
        for k in LARGE_K:
            rows = S["labels"][RUNS[0]][t]["cells"][str(k)]["rows"]
            best, ds = min(r["sim"] for r in rows), S["dataset"][t][str(k)]["best_ratio"]
            out["safest_vs_dataset"][f"{t}/K{k}"] = {"search": round(best, 4), "dataset": round(ds, 4), "search_safer": best < ds}
    out["n_search_safer"] = sum(v["search_safer"] for v in out["safest_vs_dataset"].values())

    b = np.load(c.ev_in("ch4_budget_proof.npz"), allow_pickle=True)
    fac = b["z1"] * 2 * np.pi * 1e6 * b["cap_nf"] * 1e-9            # factor of the capacitance law per placement
    m = (b["K"] == 3) & (np.abs(b["cap_nf"] - 247.0) < 0.5)
    need = 0.6 * 2 * np.pi * 1e6 * 247e-9
    out["three_decaps_0.6"] = {"n_simulated_247nF": int(m.sum()), "lowest_ratio_at_1mhz": round(float((b["z1"][m] / 0.6).min()), 4),
                               "lowest_factor_247nF": round(float(fac[m].min()), 4), "factor_needed": round(need, 4),
                               "share_all_with_factor_that_low_pct": round(100 * float((fac <= need).mean()), 2)}

    a = np.load(c.ev_in("ch4_accuracy.npz"), allow_pickle=True)
    pt = c.peak_table(a["test_pred"], a["test_sim"], f)
    i = pt[:, 1].astype(int)
    sel = c.matters(pt, tg[c.PEAK_TARGET]) & (c.region_of(f[i]) == "5-40") & (pt[:, 3] == 1)
    q = pt[sel, 4] / pt[sel, 2]
    out["low_peak_height"] = {"n_found": int(sel.sum()), "pred_over_sim_typical": round(float(np.median(q)), 3),
                              "too_low_pct": round(100 * float((q < 1).mean()), 1)}
    c.save_json(c.ev_out(OUT), out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
