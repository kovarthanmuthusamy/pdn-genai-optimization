"""Two failure cases of the surrogate, chosen by a rule rather than by eye.

What:
  case_map     a typical poor map in the decap-resonance region (5-40 MHz): among the maps of the test
               placements there (K = 4..6, first slot empty, map not flat), the one whose error of the
               hot-spot impedance is closest to the value that nine of ten such maps stay within.
  case_margin  for each target spectrum, the test placement that sets the margin of the verdict: the
               placement that fails in simulation with the lowest predicted worst-case ratio (and the
               first such placement with the first slot empty, which is the one shown).
    For both, the simulated and predicted spectra (and for case_map the maps) are stored for the figures.
Usage: python -m thesis_code.s12_evaluation.check_cases     (CONFIG-only; GPU recommended; after check_accuracy)
Outputs: ch4_cases.json and ch4_cases.npz
"""
from __future__ import annotations

import json

import numpy as np

from thesis_code.s12_evaluation import common as c

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
MAP_REGION = (5.0, 40.0)
MAP_K = (4, 5, 6)
FLAT = 1.2
# =============================================================================


def main() -> None:
    f, tg, mask = c.freqs_mhz(), c.targets(), c.board_mask()
    a = dict(np.load(c.ev_in("ch4_accuracy.npz"), allow_pickle=True))
    ids, codes, sim, pred = a["test_ids"], a["test_codes"], a["test_sim"], a["test_pred"]
    K = (codes > 0).sum(1)
    m, mi, mf = a["test_maps"], a["test_map_i"], a["test_map_f"]
    maps_of = c.maps_index()
    sur = c.Surrogate()
    res: dict = {}
    arrays: dict = {}

    # ---- case_map ---------------------------------------------------------------------------------
    sel = np.flatnonzero((mf >= MAP_REGION[0]) & (mf < MAP_REGION[1]) & np.isin(K[mi], MAP_K) & (codes[mi, 0] == 0) & (m[:, 8] >= FLAT))
    err = np.abs(m[sel, 1] - 1) * 100
    j = int(sel[np.argmin(np.abs(err - np.quantile(err, 0.9)))])
    i = int(mi[j])
    name = next(n for fq, n in maps_of[str(ids[i])] if abs(fq - mf[j]) < 1e-6)
    hs = np.load(c.data() / "raw" / "heatmap" / name)[0]
    hp = sur.maps(codes[[i]], mf[[j]])[0]
    d, ratio, cor = c.map_measures(hp, hs, mask)
    pk = c.peaks(sim[i])
    res["case_map"] = {"design_id": str(ids[i]), "K": int(K[i]), "codes": codes[i].tolist(), "mhz": float(mf[j]),
                       "n_maps_in_group": int(len(sel)), "group_error_typical_pct": c.typical(err), "group_error_9of10_pct": c.nine_of_ten(err),
                       "hot_spot_distance_mm": d, "hot_spot_ratio": ratio, "pattern_r": cor,
                       "pixels_too_high_pct": c.share(hp[mask] > hs[mask]),
                       "typical_pixel_ratio": float(np.exp(np.mean(np.log(np.maximum(hp[mask], 1e-9) / hs[mask])))),
                       "spectrum_peaks_mhz": [float(f[x]) for x in pk], "point_error_typical_pct": c.typical(np.abs(pred[i] / sim[i] - 1) * 100),
                       "nearest_peak_mhz": float(f[pk[np.argmin(np.abs(np.log(f[pk] / mf[j])))]]) if pk.size else None}
    arrays.update(case_map_sim=hs, case_map_pred=hp, case_map_spec_sim=sim[i], case_map_spec_pred=pred[i], case_map_codes=codes[i])

    # ---- case_margin ------------------------------------------------------------------------------
    res["case_margin"] = {}
    for t, zt in tg.items():
        rs, rp = c.worst_ratio(sim, zt), c.worst_ratio(pred, zt)
        fail = np.flatnonzero(rs > 1)
        order = fail[np.argsort(rp[fail])]
        i = int(order[0])
        shown = int(next(x for x in order if codes[x, 0] == 0))          # the one shown: first slot empty
        e = np.abs(rp / rs - 1) * 100
        iw = int((sim[shown] / zt).argmax()); ip = int((pred[shown] / zt).argmax())
        res["case_margin"][t] = {
            "margin_from_this_placement": float(np.ceil((1 - rp[i]) * 100 - 1e-9) / 100), "sets_margin_design_id": str(ids[i]),
            "shown_is_the_one_that_sets_it": bool(shown == i), "design_id": str(ids[shown]), "K": int(K[shown]),
            "codes": codes[shown].tolist(), "ratio_simulated": float(rs[shown]), "ratio_predicted": float(rp[shown]),
            "worst_at_mhz_simulated": float(f[iw]), "worst_at_mhz_predicted": float(f[ip]),
            "simulated_at_worst_ohm": float(sim[shown, iw]), "predicted_there_ohm": float(pred[shown, iw]),
            "n_fail": int(len(fail)), "n_fail_predicted_below_0.9": int((rp[fail] < 0.9).sum()),
            "n_fail_predicted_below_1": int((rp[fail] <= 1).sum()),
            "ratio_error_of_failing_typical_pct": c.typical(e[fail]), "ratio_error_of_failing_9of10_pct": c.nine_of_ten(e[fail])}
        arrays.update({f"case_margin_{t}_sim": sim[shown], f"case_margin_{t}_pred": pred[shown], f"case_margin_{t}_codes": codes[shown]})
    c.save_json(c.ev_out("ch4_cases.json"), res)
    np.savez_compressed(c.ev_out("ch4_cases.npz"), **arrays)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
