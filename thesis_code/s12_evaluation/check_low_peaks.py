"""What the 5-40 MHz peaks are that the surrogate misses.

What: three properties of the SIMULATED peaks of the test placements (peak definition of common:
    peaks, peak_table, matters):
    width   the width at half the prominence of ln|Z| (scipy peak_widths, rel_height 0.5), as a share
            of the peak frequency, per region; the same on |Z| in ohm as a second definition
            (width_lin), so the comparison does not hinge on the choice. The board resonances above
            180 MHz, which the surrogate finds almost always, are the reference: if the 5-40 MHz peaks
            are not narrower than those, narrowness cannot be why they are missed.
    types   the number of decap types of the placement that holds the peak. Falsifies "the peaks exist
            only for certain combinations of types" if single-type placements hold some of them.
    height  share of the peaks below 1 ohm, and the median height.
    by type count (K >= 3): the found share of the peaks that matter inside and outside the region. If
            the difference between one, two and three types is due to the region, the shares outside
            it are equal.
    Predictions are read from ch4_accuracy.npz (check_accuracy); nothing is re-run.
Usage: python -m thesis_code.s12_evaluation.check_low_peaks     (CONFIG-only; CPU, under a minute)
Outputs: ch4_low_peaks.json
"""
from __future__ import annotations

import json

import numpy as np

from thesis_code.s12_evaluation import common as c

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
REGION = "5-40"
OUT = "ch4_low_peaks.json"
# =============================================================================


def main() -> None:
    from scipy.signal import peak_widths
    d = np.load(c.ev_in("ch4_accuracy.npz"), allow_pickle=True)
    sim, pred, codes = d["test_sim"], d["test_pred"], d["test_codes"]
    f = c.freqs_mhz()
    idx = np.arange(len(f), dtype=np.float64)
    pt = c.peak_table(pred, sim, f)
    n, i = pt[:, 0].astype(int), pt[:, 1].astype(int)
    width, width_lin = np.full(len(pt), np.nan), np.full(len(pt), np.nan)
    for p in np.unique(n):
        rows = np.flatnonzero(n == p)
        for arr, z in ((width, np.log(sim[p])), (width_lin, sim[p])):
            w = peak_widths(z, i[rows], rel_height=0.5)
            arr[rows] = (np.interp(w[3], idx, f) - np.interp(w[2], idx, f)) / f[i[rows]] * 100.0
    reg = c.region_of(f[i])
    mat = c.matters(pt, c.targets()[c.PEAK_TARGET])
    mix = c.type_mix(codes)[n]
    out = {"n_test": int(len(sim)), "region": REGION, "width_pct_typical": {},
           "width_lin_pct_typical": {}, "regions": {}}
    for r in c.REGIONS:
        for name, sel in (("all", reg == r), ("matter", (reg == r) & mat)):
            if sel.any():
                out["width_pct_typical"].setdefault(r, {})[name] = round(float(np.median(width[sel])), 2)
                out["width_lin_pct_typical"].setdefault(r, {})[name] = round(float(np.median(width_lin[sel])), 2)
    for name, sel in (("all", reg == REGION), ("matter", (reg == REGION) & mat)):
        out["regions"][name] = {
            "n": int(sel.sum()), "found_pct": c.share(pt[sel, 3]),
            "by_type_count": {int(t): int((sel & (mix == t)).sum()) for t in (1, 2, 3)},
            "below_1_ohm_pct": round(100 * float((pt[sel, 2] < 1.0).mean()), 2),
            "height_median_ohm": round(float(np.median(pt[sel, 2])), 3)}
    out["placements_by_type_count"] = {int(t): int((c.type_mix(codes) == t).sum()) for t in (1, 2, 3)}
    # Split by the number of types for K >= 3: is the difference of the found share all in REGION?
    # Peaks that matter, inside and outside it, per type count.
    big = (codes > 0).sum(1)[n] >= 3
    out["by_type_count_K3plus"] = {}
    for t in (1, 2, 3):
        s = mat & big & (mix == t)
        ins, outs = s & (reg == REGION), s & (reg != REGION)
        out["by_type_count_K3plus"][int(t)] = {
            "n_matter": int(s.sum()), "found_pct": c.share(pt[s, 3]),
            "n_in_region": int(ins.sum()), "in_region_pct_of_matter": round(100 * ins.sum() / s.sum(), 2),
            "found_in_region_pct": c.share(pt[ins, 3]),
            "n_outside": int(outs.sum()), "found_outside": int(pt[outs, 3].sum())}
    c.save_json(c.ev_out(OUT), out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
