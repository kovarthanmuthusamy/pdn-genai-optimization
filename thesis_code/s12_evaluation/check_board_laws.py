"""Whether the surrogate follows the physical laws of the board.

What: three laws that the SIMULATED spectra of this board follow; the surrogate was never given them.
    Its predictions for the test placements go through the same analysis as the simulations, so each
    law is a check that needs no error measure:
      1. Total capacitance. At 1 MHz the impedance is 1 / (2 pi f C_total), times a constant close to
         one: the constant per placement from the simulated and from the predicted spectrum (median
         and the range holding 80 % of the placements), and the rank correlation of the 1 MHz
         impedance with 1 / C_total.
      2. Decaps against the board capacitance. The first peak above 40 MHz moves up with the square
         root of the number of decaps, whatever their types: median peak frequency per budget and the
         exponent of a power law fitted over the placements, simulated and predicted; and the median
         for placements of one type only (T1, T2, T3) at K = 5.
      3. Types act below 40 MHz, positions above (from check_one_decap_ecad): the typical change of a
         frequency point per region when one decap gets another type and when it is moved, simulated
         and as the surrogate predicts it.
    Null: the predictions do not reproduce a law (other constant, other exponent, a type effect where
    the simulation has none).
Usage: python -m thesis_code.s12_evaluation.check_board_laws     (CONFIG-only; CPU; after check_accuracy
       and check_one_decap_ecad)
Outputs: ch4_board_laws.json
"""
from __future__ import annotations

import json

import numpy as np

from thesis_code.s12_evaluation import common as c

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
PEAK_BAND = (40.0, 200.0)        # the first peak above 40 MHz is looked for up to 200 MHz
# =============================================================================


def first_peak(z: np.ndarray, f: np.ndarray) -> np.ndarray:
    out = np.full(len(z), np.nan)
    for n in range(len(z)):
        p = c.peaks(z[n])
        p = p[(f[p] >= PEAK_BAND[0]) & (f[p] <= PEAK_BAND[1])]
        if p.size:
            out[n] = f[p[0]]
    return out


def main() -> None:
    from scipy.stats import spearmanr
    f = c.freqs_mhz()
    a = np.load(c.ev_in("ch4_accuracy.npz"), allow_pickle=True)
    codes, K = a["test_codes"], a["test_K"]
    cap = c.total_capacitance_nf(codes)
    mix = c.type_mix(codes)
    res: dict = {"n_test": int(len(K))}
    for name, z in (("simulated", a["test_sim"]), ("predicted", a["test_pred"])):
        const = z[:, 0] * 2 * np.pi * f[0] * 1e6 * cap * 1e-9
        fp = first_peak(z, f)
        ok = np.isfinite(fp)
        expo = float(np.polyfit(np.log(K[ok]), np.log(fp[ok]), 1)[0])
        res[name] = {
            "capacitance_constant": {"median": float(np.median(const)), "p10": float(np.quantile(const, 0.1)),
                                     "p90": float(np.quantile(const, 0.9))},
            "z1_vs_inverse_capacitance_rank_correlation": float(spearmanr(z[:, 0], 1 / cap)[0]),
            "first_peak": {"placements_with_peak_pct": c.share(ok), "exponent": expo,
                           "median_mhz_per_K": {int(k): c.typical(fp[K == k]) for k in range(1, 11)},
                           "median_mhz_K5_one_type": {f"T{t}": c.typical(fp[(K == 5) & (mix == 1) & (codes == t).any(1)]) for t in (1, 2, 3)}}}
    zs, zp = a["test_sim"], a["test_pred"]
    fs, fpred = first_peak(zs, f), first_peak(zp, f)
    both = np.isfinite(fs) & np.isfinite(fpred)
    res["first_peak_agreement"] = {"n": int(both.sum()), "same_frequency_point_pct": c.share(fs[both] == fpred[both]),
                                   "frequency_error_typical_pct": c.typical(np.abs(fpred[both] / fs[both] - 1) * 100),
                                   "frequency_error_9of10_pct": c.nine_of_ten(np.abs(fpred[both] / fs[both] - 1) * 100)}
    od = c.ev_in("ch4_one_decap.json")
    o = json.loads(od.read_text()) if od.is_file() else {}
    if "point" in o.get("variants", {}).get("move", {}):          # written by check_one_decap_ecad
        res["one_decap"] = {kind: {r: {k: v[k] for k in ("change_typical_pct", "predicted_change_typical_pct", "change_9of10_pct",
                                                      "predicted_change_9of10_pct")}
                                   for r, v in o["variants"][kind]["point"].items()} for kind in ("move", "swap")}
    c.save_json(c.ev_out("ch4_board_laws.json"), res)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
