"""Effect of one decap, simulated with eCADSTAR, as the yardstick for the surrogate.

What: N_BASE_PER_K test placements per budget K = 3..10 (first slot empty) get two variants each:
      move -- one mounted decap moved to a random empty slot, same type;
      swap -- one mounted decap given another type, same slot.
    Base and variants are simulated in the same run (spectrum + one map at a frequency drawn from the
    base's own map frequencies other than 200 MHz), so the comparison is simulation against simulation
    with nothing else changed. The bare board (every slot empty) is simulated as well: spectrum and
    maps at its own resonances.
    Question: is the error of the surrogate smaller than the change that one decap causes? If it is,
    the surrogate can tell neighbouring placements apart; if the two are equal, it cannot. Both are
    stated in the same units: typical change / error of a frequency point per region, of the height of
    the peaks that matter, of the worst-case ratio; for the map the distance the hot spot moves and the
    change of its impedance. Null: error = change. Second question: does the surrogate predict the
    change itself (rank correlation of the predicted with the simulated change of the worst-case
    ratio; sign right). The variants are placements the model has not seen (checked against the
    training placements); the base re-simulations are compared with the stored dataset spectra and
    maps (the simulator is deterministic).
Usage: python -m thesis_code.s12_evaluation.check_one_decap_ecad     (CONFIG-only; resumable; needs the
       eCADSTAR host when the store is new)
Outputs: store one_decap_ecad/ (when new); ch4_one_decap.json (+ .npz with all spectra and maps)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from thesis_code.common.board import INPUTS
from thesis_code.s12_evaluation import common as c
from thesis_code.s12_evaluation import ecad

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
K_VALUES = range(3, 11)
N_BASE_PER_K = 40
SEED = 20261011
BARE_MAP_MHZ = [1.0, 10.0, 63.0, 180.0, 200.0, 240.0, 270.0, 470.0, 520.0]
MIN_CHANGE = 0.05            # sign of a change of the worst-case ratio is scored when it exceeds 5 %
STORE = "one_decap_ecad"
# =============================================================================


def prepare(st: Path) -> dict:
    rng = np.random.default_rng(SEED)
    test = c.split_ids("test")
    maps_of = c.maps_index()
    codes = dict(zip(test, c.codes_of(test)))
    groups, bases = [], []
    for k in K_VALUES:
        pool = [d for d in test if (codes[d] > 0).sum() == k and codes[d][0] == 0]
        for d in rng.choice(pool, size=min(N_BASE_PER_K, len(pool)), replace=False):
            cd = codes[d]
            others = [(f, n) for f, n in maps_of[d] if abs(f - c.F0_MHZ) > 1e-6]
            f, sample = others[int(rng.integers(len(others)))]
            b = len(bases)
            bases.append({"design_id": str(d), "K": int(k), "codes": cd.tolist(), "map_mhz": f, "map_sample": sample})
            groups.append({"kind": "base", "base": b, "codes": cd.tolist(), "map_mhz": f})
            mounted, empty = np.flatnonzero(cd > 0), np.flatnonzero(cd == 0)
            empty = empty[empty != 0]                       # the first slot stays empty
            src = int(rng.choice(mounted))
            mv = cd.copy(); mv[int(rng.choice(empty))] = cd[src]; mv[src] = 0
            groups.append({"kind": "move", "base": b, "codes": mv.tolist(), "map_mhz": f, "moved_from": src,
                           "moved_to": int(np.flatnonzero((mv > 0) & (cd == 0))[0])})
            sw = cd.copy(); i = int(rng.choice(mounted))
            sw[i] = int(rng.choice([t for t in (1, 2, 3) if t != cd[i]]))
            groups.append({"kind": "swap", "base": b, "codes": sw.tolist(), "map_mhz": f, "slot": i,
                           "type_from": int(cd[i]), "type_to": int(sw[i])})
    bare = [0] * c.N_SLOTS
    spectra = [{"kind": "bare", "codes": bare}] + [{k: g[k] for k in ("kind", "base", "codes")} for g in groups]
    maps = [{"kind": "bare", "codes": bare, "map_mhz": f} for f in BARE_MAP_MHZ] + \
           [{k: g[k] for k in ("kind", "base", "codes", "map_mhz")} for g in groups]
    man = {"dataset": c.data().name, "seed": SEED, "bases": bases, "groups": groups,
           "spectra": ecad.assign(spectra), "maps": ecad.assign(maps)}
    c.save_json(ecad.writable(st) / "manifest.json", man)
    print(f"{len(bases)} bases -> {len(spectra)} spectra, {len(maps)} maps", flush=True)
    return man


def analyze(st: Path, man: dict) -> dict:
    from scipy.stats import spearmanr
    mask, f, tg = c.board_mask(), c.freqs_mhz(), c.targets()
    Z, H = ecad.read_spectra(st, man["spectra"]), ecad.read_maps(st, man["maps"])
    reg = {r: (f >= lo) & (f < hi) for r, (lo, hi) in c.REGIONS.items()}
    gi = {(r["kind"], r.get("base")): i for i, r in enumerate(man["spectra"])}
    mi = {(r["kind"], r.get("base")): i for i, r in enumerate(man["maps"]) if r["kind"] != "bare"}
    nb = len(man["bases"])
    K = np.array([b["K"] for b in man["bases"]])
    cb = np.array([b["codes"] for b in man["bases"]])
    zb = np.stack([Z[gi[("base", b)]] for b in range(nb)])
    hb = np.stack([H[mi[("base", b)]] for b in range(nb)])
    mhz = np.array([b["map_mhz"] for b in man["bases"]])
    stored = c.spectra_of([b["design_id"] for b in man["bases"]])
    hstored = np.stack([np.load(c.data() / "raw" / "heatmap" / b["map_sample"])[0] for b in man["bases"]])
    train = {x.tobytes() for x in c.codes_of(c.split_ids("train"))}
    sur = c.Surrogate()
    pb, pmb = sur.spectra(cb), sur.maps(cb, mhz)
    zbare = Z[gi[("bare", None)]]
    out = {"n_bases": nb,
           "bare": {"z_1mhz": float(zbare[0]), "peaks": [(float(f[i]), float(zbare[i])) for i in c.peaks(zbare)],
                    "max_rel_diff_vs_stored": float(np.max(np.abs(zbare / np.load(INPUTS / 'bare_impedance.npy').reshape(-1) - 1)))},
           "reproducibility": {"spectrum_max_rel_diff": float(np.max(np.abs(zb / stored - 1))),
                               "map_max_rel_diff": float(np.max(np.abs(hb[:, mask] / hstored[:, mask] - 1)))},
           "variants": {}}
    arrays = {"Z": Z, "H": H, "K": K, "pred_base": pb, "pred_base_maps": pmb, "map_mhz_base": mhz}

    def peak_change(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> dict:
        """Peaks of ``b`` that matter for target ``t``, looked up in ``a``: kept share and change of the height."""
        pt = c.peak_table(a, b, f)
        pt = pt[c.matters(pt, t)]
        return {"n": int(len(pt)), "kept_pct": c.share(pt[:, 3]),
                "height_change_typical_pct": c.typical(np.abs(pt[:, 4] / pt[:, 2] - 1) * 100),
                "height_change_9of10_pct": c.nine_of_ten(np.abs(pt[:, 4] / pt[:, 2] - 1) * 100)}

    for kind in ("move", "swap"):
        cv = np.array([man["spectra"][gi[(kind, b)]]["codes"] for b in range(nb)])
        zv = np.stack([Z[gi[(kind, b)]] for b in range(nb)])
        hv = np.stack([H[mi[(kind, b)]] for b in range(nb)])
        pv, pmv = sur.spectra(cv), sur.maps(cv, mhz)
        ch, er = np.abs(zv / zb - 1) * 100, np.abs(pv / zv - 1) * 100            # change by one decap, error of the surrogate
        pch = np.abs(pv / pb - 1) * 100                                          # the change as the surrogate predicts it
        mch = np.array([c.map_measures(hv[b], hb[b], mask) for b in range(nb)])   # variant against base (simulated)
        mer = np.array([c.map_measures(pmv[b], hv[b], mask) for b in range(nb)])  # prediction against the simulated variant
        v = {"n_in_training_set": int(sum(x.tobytes() in train for x in cv)),
             "point": {r: {"change_typical_pct": c.typical(ch[:, m]), "change_9of10_pct": c.nine_of_ten(ch[:, m]),
                           "predicted_change_typical_pct": c.typical(pch[:, m]), "predicted_change_9of10_pct": c.nine_of_ten(pch[:, m]),
                           "error_typical_pct": c.typical(er[:, m]), "error_9of10_pct": c.nine_of_ten(er[:, m])}
                       for r, m in {**reg, "all": np.ones(len(f), bool)}.items()},
             "largest_change_typical_pct": c.typical(ch.max(1)),
             "map": {"hot_spot_moved_typical_mm": c.typical(mch[:, 0]), "hot_spot_moved_9of10_mm": c.nine_of_ten(mch[:, 0]),
                     "hot_spot_change_typical_pct": c.typical(np.abs(mch[:, 1] - 1) * 100),
                     "error": c.map_summary(mer)},
             "targets": {}}
        for t, zt in tg.items():
            rb, rv, qb, qv = (c.worst_ratio(x, zt) for x in (zb, zv, pb, pv))
            d_sim, d_pred = rv / rb - 1, qv / qb - 1
            big = np.abs(d_sim) > MIN_CHANGE
            v["targets"][t] = {
                "peaks_change": peak_change(zv, zb, zt), "peaks_error": c.spectrum_summary(pv, zv, f, zt)["peaks_that_matter"],
                "ratio_change_typical_pct": c.typical(np.abs(d_sim) * 100), "ratio_change_9of10_pct": c.nine_of_ten(np.abs(d_sim) * 100),
                "ratio_error_typical_pct": c.typical(np.abs(qv / rv - 1) * 100), "ratio_error_9of10_pct": c.nine_of_ten(np.abs(qv / rv - 1) * 100),
                "verdict_flips_pct": c.share((rv <= 1) != (rb <= 1)),
                "flip_predicted_pct": c.share(((qv <= 1) != (qb <= 1)) == ((rv <= 1) != (rb <= 1))),
                "change_rank_correlation": float(spearmanr(d_sim, d_pred)[0]),
                "n_change_above_min": int(big.sum()), "sign_right_pct": c.share(np.sign(d_pred[big]) == np.sign(d_sim[big]))}
        out["variants"][kind] = v
        arrays.update({f"pred_{kind}": pv, f"pred_{kind}_maps": pmv, f"codes_{kind}": cv})
    c.save_json(c.ev_out("ch4_one_decap.json"), out)
    np.savez_compressed(c.ev_out("ch4_one_decap.npz"), **arrays, codes_base=cb, bare_map_mhz=np.array(BARE_MAP_MHZ),
                        map_mhz=np.array([r["map_mhz"] for r in man["maps"]]),
                        map_kind=np.array([r["kind"] for r in man["maps"]]),
                        spec_kind=np.array([r["kind"] for r in man["spectra"]]),
                        spec_base=np.array([-1 if r.get("base") is None else r["base"] for r in man["spectra"]]),
                        map_base=np.array([-1 if r.get("base") is None else r["base"] for r in man["maps"]]))
    print(json.dumps(out, indent=1))
    return out


def main() -> None:
    st = ecad.store(STORE)
    man_p = st / "manifest.json"
    man = json.loads(man_p.read_text()) if man_p.is_file() else prepare(st)
    ecad.simulate(st, man["spectra"], man["maps"])
    analyze(st, man)


if __name__ == "__main__":
    main()
