"""Where the first board peak goes beyond the trained budgets.

What: placements with 11 and 12 decaps were predicted with the budget input held at 10
    (check_beyond_range_ecad). If the surrogate treats them as placements of ten decaps, the peak law
    of the board (f_K = f_1 sqrt(K)) says their first peak above 40 MHz is predicted too low by the
    factor sqrt(10 / K). Measured: the first peak above 40 MHz (check_board_laws.first_peak) of the
    simulated and of the predicted spectrum, and the typical ratio predicted / simulated per budget,
    set against the law's value. The frequency error of all peaks that matter mixes this peak with the
    board resonances, which do not move with the budget, so it cannot test the law.
Usage: python -m thesis_code.s12_evaluation.check_beyond_peak     (CONFIG-only; CPU, seconds)
Outputs: ch4_beyond_peak.json
"""
from __future__ import annotations

import json

import numpy as np

from thesis_code.s12_evaluation import common as c
from thesis_code.s12_evaluation.check_board_laws import first_peak

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
BUDGETS = (11, 12)
K_INPUT = 10                         # the budget input the placements were predicted with
OUT = "ch4_beyond_peak.json"
# =============================================================================


def main() -> None:
    d = np.load(c.ev_in("ch4_beyond_range.npz"), allow_pickle=True)
    f = c.freqs_mhz()
    out: dict = {"k_input": K_INPUT, "budgets": {}}
    for k in BUDGETS:
        m = d["K"] == k
        fs, fp = first_peak(d["Z"][m], f), first_peak(d["pred_clamped"][m], f)
        ok = np.isfinite(fs) & np.isfinite(fp)
        q = fp[ok] / fs[ok]
        out["budgets"][k] = {"n": int(ok.sum()), "first_peak_sim_median_mhz": float(np.median(fs[ok])),
                             "first_peak_pred_median_mhz": float(np.median(fp[ok])),
                             "pred_over_sim_typical": round(float(np.median(q)), 4),
                             "too_low_typical_pct": round(100 * (1 - float(np.median(q))), 2),
                             "law_too_low_pct": round(100 * (1 - np.sqrt(K_INPUT / k)), 2)}
    c.save_json(c.ev_out(OUT), out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
