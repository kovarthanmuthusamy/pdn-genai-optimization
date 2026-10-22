"""Where on the board a proposed placement is closest to its limit: maps against eCADSTAR.

What: for a proposed placement the designer wants to see where on the board the impedance is high at
    the frequency that limits the placement. For every placement that the main search run (label s0)
    proposed and that passes in eCADSTAR:
      f*   the frequency with the highest predicted ratio to the target spectrum, from the spectrum
           the surrogate predicts for the placement alone;
      map  the surrogate's map at f*, and the eCADSTAR map at the same f*.
    Measures as for the test placements: distance between the predicted and the simulated hot spot in
    mm, simulated impedance at the predicted hot spot against the true maximum, error of the hot-spot
    impedance, pattern right. Also whether f* is the right frequency: the simulated spectrum's own
    limiting frequency within 10 % of f*.
    Reference: the maps of the test placements in the same frequency region (ch4_accuracy.npz).
    Proposed placements are chosen by a search, so clearly worse maps here than there would mean that
    the search steers toward placements whose maps the surrogate predicts poorly. Null: no difference.
Usage: python -m thesis_code.s12_evaluation.check_search_maps_ecad     (CONFIG-only; resumable; GPU
       recommended + eCADSTAR when the store is new; after check_search_ecad)
Outputs: store search_maps_ecad/ (when new); ch4_search_maps.json (+ .npz)
"""
from __future__ import annotations

import json

import numpy as np

from thesis_code.s12_evaluation import common as c
from thesis_code.s12_evaluation import ecad

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
LABEL = "s0"
SEARCH_FILE = "ch4_search.json"               # proposals with their simulated ratio (check_search_ecad)
MAPS_FILE = "ch4_search_maps.json"            # result; the .npz of the same name holds the maps
STORE = "search_maps_ecad"
SPECTRA_STORE = "search_ecad"
# =============================================================================


def main() -> None:
    f, tg, mask = c.freqs_mhz(), c.targets(), c.board_mask()
    res = json.loads(c.ev_in(SEARCH_FILE).read_text())["labels"][LABEL]
    rows = [{"target": t, "K": int(k), "codes": r["codes"], "safe": r["recheck"] <= 1 - cell_t["margin"], "sim_ratio": r["sim"]}
            for t, cell_t in res.items() for k, cell in cell_t["cells"].items() for r in cell["rows"] if r["sim"] <= 1]
    codes = np.array([r["codes"] for r in rows], dtype=np.int64)
    sur = c.Surrogate()
    zp = sur.spectra(codes)
    d = np.load(ecad.store(SPECTRA_STORE) / "spectra.npz")
    sim = {x.tobytes(): z for x, z in zip(d["codes"], d["values"])}
    zs = np.stack([sim[x.tobytes()] for x in codes])
    zt = np.stack([tg[r["target"]] for r in rows])
    i_pred, i_sim = (zp / zt).argmax(1), (zs / zt).argmax(1)
    fstar = f[i_pred]
    pm = sur.maps(codes, fstar)
    H = ecad.maps_cached(ecad.store(STORE), codes, fstar)
    m = np.array([c.map_measures(pm[i], H[i], mask) for i in range(len(rows))])
    at = np.array([H[i][np.unravel_index(np.where(mask, pm[i], -np.inf).argmax(), mask.shape)] / H[i][mask].max() for i in range(len(rows))])
    same_f = np.abs(np.log(f[i_sim] / fstar)) <= np.log(1.10)
    safe = np.array([r["safe"] for r in rows]); tt = np.array([r["target"] for r in rows]); reg = c.region_of(fstar)

    def block(sel) -> dict:
        return {**c.map_summary(m[sel]), "within_10mm_pct": c.share(m[sel, 0] <= 10.2),
                "at_predicted_hot_spot_90pct_of_max_pct": c.share(at[sel] >= 0.9),
                "at_predicted_hot_spot_typical_pct_of_max": c.typical(at[sel] * 100),
                "limiting_frequency_right_pct": c.share(same_f[sel])}
    out = {"label": LABEL, "n": len(rows), "all": block(np.ones(len(rows), bool)), "safe": block(safe),
           "targets": {t: block(tt == t) for t in tg}, "regions": {r: {"n": int((reg == r).sum()), **block(reg == r)} for r in c.REGIONS if (reg == r).any()},
           "fstar_mhz": {"typical": c.typical(fstar), "min": float(fstar.min()), "max": float(fstar.max())}}
    a = np.load(c.ev_in("ch4_accuracy.npz"), allow_pickle=True)
    tm, tf = a["test_maps"], a["test_map_f"]
    treg = c.region_of(tf)
    ok = np.isfinite(tm[:, 0])
    out["test_placements_same_regions"] = {r: {**c.map_summary(tm[ok & (treg == r), :3]), "within_10mm_pct": c.share(tm[ok & (treg == r), 0] <= 10.2),
                                               "at_predicted_hot_spot_90pct_of_max_pct": c.share(tm[ok & (treg == r), 3] >= 0.9)}
                                           for r in c.REGIONS}
    c.save_json(c.ev_out(MAPS_FILE), out)
    np.savez_compressed(c.ev_out(MAPS_FILE).with_suffix(".npz"), codes=codes, fstar=fstar, pred=pm.astype(np.float32), sim=H, measures=m,
                        at_pred=at, safe=safe, target=tt, K=np.array([r["K"] for r in rows]), spec_pred=zp, spec_sim=zs)
    print(json.dumps({k: out[k] for k in ("n", "all", "safe", "regions", "test_placements_same_regions")}, indent=1))


if __name__ == "__main__":
    main()
