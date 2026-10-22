"""Why the closest-placement lookup is poor even one decap away.

What: for the test placements that differ from their closest training placement in a single decap,
    the lookup of check_accuracy is compared with the surrogate. Two questions, on the saved predictions:
    budget   the distance-1 group split by the number of decaps K. One decap is a larger part of a small
             placement, so if small budgets dominate the group they, not the distance, set its error.
    types    the share of the 5-40 MHz peaks that matter which the lookup finds, split by whether the
             closest training placement has the same number of decaps of each type. If the lookup finds
             these peaks because its placement has similar types, the share is higher when the counts match.
    The lookup indices are recomputed with the seed of check_accuracy and checked against the saved
    lookup spectra before use.
Usage: python -m thesis_code.s12_evaluation.check_lookup_near     (CONFIG-only; CPU, about a minute)
Outputs: ch4_lookup_near.json
"""
from __future__ import annotations

import json

import numpy as np

from thesis_code.s12_evaluation import common as c

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
SEED = 20261011                 # = check_accuracy.SEED (tie-breaking of the lookup)
REGION = "5-40"
OUT = "ch4_lookup_near.json"
# =============================================================================


def main() -> None:
    d = np.load(c.ev_in("ch4_accuracy.npz"), allow_pickle=True)
    sim, pred, lk, codes, dist, K = (d[k] for k in ("test_sim", "test_pred", "test_lookup", "test_codes",
                                                   "test_dist", "test_K"))
    e_lk, e_sur = np.abs(lk / sim - 1) * 100, np.abs(pred / sim - 1) * 100
    one = dist == 1
    out = {"n_distance_1": int(one.sum()), "distance_1_by_K": {}}
    for k in range(1, 11):
        m = one & (K == k)
        if m.any():
            out["distance_1_by_K"][k] = {"n": int(m.sum()), "lookup_point_typical_pct": round(float(np.median(e_lk[m])), 2),
                                         "surrogate_point_typical_pct": round(float(np.median(e_sur[m])), 2)}

    c_tr = c.codes_of(c.split_ids("train"))
    nn, _ = c.closest_training(codes, c_tr, SEED)
    assert np.allclose(c.spectra_of(c.split_ids("train"))[nn], lk), "lookup indices do not reproduce the saved lookup"
    f = c.freqs_mhz()
    pt = c.peak_table(lk, sim, f)
    n, i = pt[:, 0].astype(int), pt[:, 1].astype(int)
    sel = c.matters(pt, c.targets()[c.PEAK_TARGET]) & (c.region_of(f[i]) == REGION)
    counts = lambda x: np.stack([(x == t).sum(1) for t in (1, 2, 3)], 1)  # noqa: E731
    same = (counts(codes) == counts(c_tr[nn])).all(1)[n]
    out["lookup_region_peaks"] = {"region": REGION, "n": int(sel.sum()), "found_pct": c.share(pt[sel, 3])}
    for name, m in (("same_type_counts", same), ("other_type_counts", ~same)):
        out["lookup_region_peaks"][name] = {"n": int((sel & m).sum()), "found_pct": c.share(pt[sel & m, 3])}
    c.save_json(c.ev_out(OUT), out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
