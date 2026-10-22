"""Accuracy of the surrogate from the placement alone, against eCADSTAR.

What: the surrogate is given only a placement it has not seen and predicts the spectrum and the maps;
    both are compared with the eCADSTAR simulation of the same placement.
      spectrum  peaks found (peaks that matter and all peaks), error of the peak height, shape of the
                curve right, error of a frequency point. Dips are not scored.
      maps      distance between the predicted and the simulated hot spot in mm, error of the hot-spot
                impedance, pattern right; plus the simulated impedance at the predicted hot spot
                against the true maximum (what is lost by looking where the surrogate points).
      verdict   pass/fail against the three target spectra, and the error of the worst-case ratio.
    Reference with no model -- closest-placement lookup: the simulation of the most similar training
    placement (same number of decaps; distance = decaps not held in the same slot with the same type;
    ties broken at random) is used as the "prediction". For a map at frequency f the lookup takes the
    closest training placement that has a map within +-5 % of f (10 / 20 % if none). The lookup has
    the same data and no model, so equal numbers would mean the surrogate adds nothing to a database.
    Null: surrogate = lookup.
    Sets: test placements (all eleven maps each) and the 480 independent placements (eleven fixed
    frequencies). Splits: frequency region, budget K, number of decap types, distance to the closest
    training placement.
Usage: python -m thesis_code.s12_evaluation.check_accuracy     (CONFIG-only; GPU recommended)
Outputs: ch4_accuracy.json and ch4_accuracy.npz (per placement / per map arrays, read by later checks)
"""
from __future__ import annotations

import json

import numpy as np

from thesis_code.common import settings
from thesis_code.s12_evaluation import common as c

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
PEAK_TARGET = c.PEAK_TARGET              # peaks that matter are taken against the tightest target spectrum
LOOKUP_FREQ_TOL = (1.05, 1.10, 1.20)
FLAT = 1.2                               # a simulated map is "flat" when its maximum is below FLAT x its median
EXAMPLE_K = 5                            # budget of the example placement shown in the figures
SEED = 20261011
# =============================================================================


def map_row(p: np.ndarray, r: np.ndarray, mask: np.ndarray) -> tuple:
    """map_measures + simulated impedance at the predicted hot spot / simulated maximum + both hot-spot pixels."""
    d, ratio, cor = c.map_measures(p, r, mask)
    pi, pj = np.unravel_index(np.where(mask, p, -np.inf).argmax(), p.shape)
    ri, rj = np.unravel_index(np.where(mask, r, -np.inf).argmax(), r.shape)
    return d, ratio, cor, float(r[pi, pj] / r[ri, rj]), pi, pj, ri, rj


def spec_block(pred, sim, f, tg, groups: dict) -> dict:
    """Spectrum measures for every group of placements (name -> bool mask), per region and per target."""
    pt = c.peak_table(pred, sim, f)
    reg = {r: (f >= lo) & (f < hi) for r, (lo, hi) in c.REGIONS.items()}
    out = {}
    for name, m in groups.items():
        if not m.any():
            continue
        s = c.spectrum_summary(pred, sim, f, tg[PEAK_TARGET], pt=pt, placements=m)
        s["regions"] = {r: c.spectrum_summary(pred, sim, f, tg[PEAK_TARGET], pt=pt, placements=m, freq_sel=fs)
                        for r, fs in reg.items()} if name == "all" else None
        s["targets"] = {}
        for t, zt in tg.items():
            rp, rs = c.worst_ratio(pred[m], zt), c.worst_ratio(sim[m], zt)
            e = np.abs(rp / rs - 1) * 100
            s["targets"][t] = {"verdict_agreement_pct": c.share((rp <= 1) == (rs <= 1)),
                               "n_pass_simulated": int((rs <= 1).sum()), "n_fail_simulated": int((rs > 1).sum()),
                               "false_pass_pct": c.share((rp <= 1)[rs > 1]),       # of the placements that fail (Eq. verdict_rates, m = 0)
                               "false_fail_pct": c.share((rp > 1)[rs <= 1]),       # of the placements that pass
                               "passing_simulated_pct": c.share(rs <= 1),
                               "ratio_error_typical_pct": c.typical(e), "ratio_error_9of10_pct": c.nine_of_ten(e),
                               "ratio_signed_typical_pct": c.typical((rp / rs - 1) * 100),
                               "peaks_that_matter": c.peak_stats(pt[m[pt[:, 0].astype(int)] & c.matters(pt, zt)], f)}
        out[name] = s
    return out


def map_block(rows: np.ndarray, groups: dict) -> dict:
    out = {}
    for name, m in groups.items():
        if m.any():
            out[name] = {**c.map_summary(rows[m, :3]), "as_hot_as_true_pct": c.share(rows[m, 3][np.isfinite(rows[m, 3])] >= c.AS_HOT),
                         "at_predicted_hot_spot_typical_pct_of_max": c.typical(rows[m, 3] * 100),
                         "at_predicted_hot_spot_1of10_below_pct_of_max": float(np.quantile(rows[m, 3], 0.1) * 100)}
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    f, tg, mask = c.freqs_mhz(), c.targets(), c.board_mask()
    train, test = c.split_ids("train"), c.split_ids("test")
    maps_of = c.maps_index()
    c_tr, z_tr = c.codes_of(train), c.spectra_of(train)
    k_tr = (c_tr > 0).sum(1)
    c_te, z_te = c.codes_of(test), c.spectra_of(test)
    k_te = (c_te > 0).sum(1)
    tr_maps = [(i, fq, n) for i, d in enumerate(train) for fq, n in maps_of[d]]
    tr_f = np.array([x[1] for x in tr_maps]); tr_i = np.array([x[0] for x in tr_maps])
    print(f"{len(test):,} test placements, {len(train):,} training placements / {len(tr_maps):,} maps", flush=True)
    sur = c.Surrogate()

    def lookup_map(code: np.ndarray, fq: float) -> np.ndarray | None:
        k = int((code > 0).sum())
        for tol in LOOKUP_FREQ_TOL:
            sel = np.flatnonzero((tr_f >= fq / tol) & (tr_f <= fq * tol) & (k_tr[tr_i] == k))
            if sel.size:
                D = c.distances(code[None], c_tr[tr_i[sel]])[0]
                return np.load(c.data() / "raw" / "heatmap" / tr_maps[int(rng.choice(sel[D == D.min()]))][2])[0]
        return None

    def run_set(codes, sims, map_items, sim_map) -> dict:
        """map_items: (placement index, MHz); sim_map(j) -> simulated map of item j."""
        pred = sur.spectra(codes)
        nn, dist = c.closest_training(codes, c_tr, SEED)
        mi = np.array([i for i, _ in map_items]); mf = np.array([x for _, x in map_items])
        rows, lrows = [], []
        for s in range(0, len(map_items), 2048):
            pm = sur.maps(codes[mi[s:s + 2048]], mf[s:s + 2048])
            for j in range(len(pm)):
                r = sim_map(s + j)
                if r is None:
                    rows.append((np.nan,) * 9); lrows.append((np.nan,) * 4); continue
                rows.append((*map_row(pm[j], r, mask), float(r[mask].max() / np.median(r[mask]))))
                lk = lookup_map(codes[mi[s + j]], float(mf[s + j]))
                lrows.append(map_row(lk, r, mask)[:4] if lk is not None else (np.nan,) * 4)
            print(f"  maps {min(s + 2048, len(map_items)):,}/{len(map_items):,}", flush=True)
        return {"pred": pred, "lookup": z_tr[nn], "dist": dist, "map_i": mi, "map_f": mf,
                "maps": np.array(rows, dtype=np.float64), "maps_lookup": np.array(lrows, dtype=np.float64)}

    # ---- test placements ---------------------------------------------------------------------------
    items = [(i, fq, n) for i, d in enumerate(test) for fq, n in maps_of[d]]
    te = run_set(c_te, z_te, [(i, fq) for i, fq, _ in items],
                 lambda j: np.load(c.data() / "raw" / "heatmap" / items[j][2])[0])

    # ---- independent placements --------------------------------------------------------------------
    ind = settings.INDEPENDENT_DIR
    man = json.loads((ind / "unseen_manifest.json").read_text())
    c_in = np.array([r["type_codes"] for r in man["rows"]], dtype=np.int64)
    z_in = np.load(ind / "pred" / "impedance_real_ohm.npy").reshape(len(c_in), -1).astype(np.float64)
    sim_in = {float(d.name[5:-3]): np.load(d / "real_ohm.npy") for d in sorted((ind / "pred").glob("freq_*MHz"))}
    it_in = [(i, fq) for fq in sorted(sim_in) for i in range(len(c_in))]
    def _sim_in(j):
        m = sim_in[it_in[j][1]][it_in[j][0]]
        return m.astype(np.float64) if np.isfinite(m).all() else None
    ie = run_set(c_in, z_in, it_in, _sim_in)
    seen = {x.tobytes() for x in c_tr}

    # ---- summaries ---------------------------------------------------------------------------------
    def groups_of(codes, dist) -> dict:
        k, mix = (codes > 0).sum(1), c.type_mix(codes)
        g = {"all": np.ones(len(codes), bool)}
        g.update({f"K={x}": k == x for x in range(1, 11)})
        g.update({f"types={x}": (mix == x) & (k >= 3) for x in (1, 2, 3)})
        g.update({f"distance={x}": dist == x for x in np.unique(dist)})
        return g

    def summarize(codes, sims, d) -> dict:
        g = groups_of(codes, d["dist"])
        k = (codes > 0).sum(1)
        ok = np.isfinite(d["maps"][:, 0])
        reg = c.region_of(d["map_f"])
        mg = {"all": ok, "not_flat": ok & (d["maps"][:, 8] >= FLAT), "at_200": ok & (d["map_f"] == c.F0_MHZ),
              **{f"region {r}": ok & (reg == r) for r in c.REGIONS},
              **{f"region {r} not_flat": ok & (reg == r) & (d["maps"][:, 8] >= FLAT) for r in c.REGIONS},
              **{name: ok & m[d["map_i"]] for name, m in g.items() if name != "all"}}
        return {"n_placements": int(len(codes)), "n_maps": int(ok.sum()),
                "n_in_training_set": int(sum(x.tobytes() in seen for x in codes)),
                "distance_per_K": {int(x): {"typical": c.typical(d["dist"][k == x]), "min": int(d["dist"][k == x].min()),
                                            "max": int(d["dist"][k == x].max())} for x in np.unique(k)},
                "flat_maps_pct": c.share(d["maps"][ok, 8] < FLAT),
                "flat_maps_pct_per_region": {r: c.share(d["maps"][ok & (reg == r), 8] < FLAT) for r in c.REGIONS},
                "spectrum": {"surrogate": spec_block(d["pred"], sims, f, tg, g), "lookup": spec_block(d["lookup"], sims, f, tg, g)},
                "maps": {"surrogate": map_block(d["maps"], mg), "lookup": map_block(d["maps_lookup"], mg)}}

    # ---- example placement for the figures: a typical test placement ------------------------------
    # K = EXAMPLE_K, all three types, first slot empty; the one whose typical point error is the median of that group
    grp = np.flatnonzero((k_te == EXAMPLE_K) & (c.type_mix(c_te) == 3) & (c_te[:, 0] == 0))
    perr = np.median(np.abs(te["pred"][grp] / z_te[grp] - 1), axis=1)
    ex = int(grp[np.argsort(perr)[len(perr) // 2]])
    ex_items = sorted((fq, n) for fq, n in maps_of[test[ex]])
    ex_f = np.array([fq for fq, _ in ex_items])
    ex_sim = np.stack([np.load(c.data() / "raw" / "heatmap" / n)[0] for _, n in ex_items])
    ex_pred = sur.maps(np.repeat(c_te[[ex]], len(ex_f), 0), ex_f)
    print(f"example placement: {test[ex]} (typical point error {np.median(perr) * 100:.1f} %), maps at {ex_f.tolist()}", flush=True)

    res = {"dataset": c.data().name, "peak_target": PEAK_TARGET, "example_placement": test[ex],
           "definitions": {"peak_prominence_ln": c.PEAK_PROMINENCE, "peak_matters": c.PEAK_MATTERS, "peak_freq_tol": c.PEAK_FREQ_TOL,
                           "shape_r": c.SHAPE_R, "pattern_r": c.PATTERN_R, "pixel_mm": [c.PIX_X, c.PIX_Y], "flat": FLAT},
           "test": summarize(c_te, z_te, te), "independent": summarize(c_in, z_in, ie)}
    c.save_json(c.ev_out("ch4_accuracy.json"), res)
    np.savez_compressed(
        c.ev_out("ch4_accuracy.npz"), test_codes=c_te, test_K=k_te, test_sim=z_te, test_pred=te["pred"], test_lookup=te["lookup"],
        test_dist=te["dist"], test_map_i=te["map_i"], test_map_f=te["map_f"], test_maps=te["maps"], test_maps_lookup=te["maps_lookup"],
        ind_codes=c_in, ind_sim=z_in, ind_pred=ie["pred"], ind_lookup=ie["lookup"], ind_dist=ie["dist"], ind_map_i=ie["map_i"],
        ind_map_f=ie["map_f"], ind_maps=ie["maps"], ind_maps_lookup=ie["maps_lookup"],
        example_index=ex, example_map_f=ex_f, example_maps_sim=ex_sim, example_maps_pred=ex_pred,
        test_ids=np.array(test), map_columns=np.array(["distance_mm", "hot_spot_ratio", "pattern_r", "at_predicted_hot_spot",
                                                       "pred_i", "pred_j", "sim_i", "sim_j", "max_over_median"]))
    short = {s: {"spectrum": {m: {k: v for k, v in res[s]["spectrum"][m]["all"].items() if k not in ("regions", "targets")}
                              for m in ("surrogate", "lookup")},
                 "maps": {m: res[s]["maps"][m]["all"] for m in ("surrogate", "lookup")}} for s in ("test", "independent")}
    print(json.dumps(short, indent=1))


if __name__ == "__main__":
    main()
