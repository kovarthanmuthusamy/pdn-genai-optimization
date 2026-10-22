"""The smallest budget that can meet each target spectrum at all, without a model.

What: three model-free facts against which the search is judged.
  1. Capacitance bound. At 1 MHz the board with decaps is a capacitor, so the impedance there follows
     from the total capacitance of the mounted decaps alone. Measured on every simulated placement
     (the dataset and its test pool): |Z(1 MHz)| x 2 pi f C_total. A target that asks for T ohm at
     1 MHz therefore needs a minimum total capacitance, and with the largest type (T1, 100 nF) a
     minimum number of decaps. Checked directly: the smallest total capacitance among all simulated
     placements whose 1 MHz impedance is at or below the target, and the smallest number of decaps.
  2. Proof by simulation where the bound is not tight. For the 0.8 ohm target the bound allows two
     decaps, both T1. All 1,326 placements of two T1 decaps are simulated (dataset and test pool);
     those that mount the first slot are simulated again here with that slot first in the batch. If
     none of the 1,326 passes, three decaps is the true minimum.
  3. Rule of thumb of a designer: the largest decaps as close to the IC as possible. For K = 2..10:
     T1 decaps in the K slots closest to the IC port, simulated; smallest K that passes each target.
  Also: the smallest budget with a passing placement among all simulated placements, and the share of
  the dataset's placements of each budget that pass. What would falsify "the search finds the smallest
  possible budget": a simulated placement that passes with fewer decaps than the search's.
Usage: python -m thesis_code.s12_evaluation.check_budget_proof_ecad     (CONFIG-only; 60 simulations,
       about 1 min of eCADSTAR when the store is new)
Outputs: store budget_proof_ecad/ (when new); ch4_budget_proof.json (+ .npz)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from thesis_code.s12_evaluation import common as c
from thesis_code.s12_evaluation import ecad

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
RULE_K = range(2, 11)
STORE = "budget_proof_ecad"
# =============================================================================


def _index(root: Path) -> list[dict]:
    with (root / "index_layouts.csv").open() as fh:
        return list(csv.DictReader(fh))


def prepare(st: Path) -> dict:
    xy, ic = c.slot_xy_mm()
    order = np.argsort(np.hypot(*(xy - ic).T), kind="stable")
    rows = []
    for k in RULE_K:
        cd = np.zeros(c.N_SLOTS, dtype=int); cd[order[:k]] = 1
        rows.append({"kind": "rule", "K": int(k), "codes": cd.tolist()})
    for r in _index(c.data()) + _index(c.data() / "test_pool"):
        if int(r["K"]) == 2 and int(r["n_100nF"]) == 2 and r["type_codes"][0] != "0":
            rows.append({"kind": "pair_first_slot", "design_id": r["design_id"], "codes": [int(x) for x in r["type_codes"]]})
    man = {"closest_slots": [int(i) + 1 for i in order[:10]], "distance_mm": [float(np.hypot(*(xy[i] - ic))) for i in order[:10]],
           "spectra": ecad.assign(rows)}
    c.save_json(ecad.writable(st) / "manifest.json", man)
    print(f"{len(rows)} spectra to simulate", flush=True)
    return man


def analyze(st: Path, man: dict) -> dict:
    f, tg = c.freqs_mhz(), c.targets()
    Z = ecad.read_spectra(st, man["spectra"])
    resim = {r["design_id"]: Z[i] for i, r in enumerate(man["spectra"]) if r["kind"] == "pair_first_slot"}
    codes, spec, K, src = [], [], [], []
    for name, root in (("dataset", c.data()), ("pool", c.data() / "test_pool")):
        idx = _index(root)
        ids = [r["design_id"] for r in idx]
        codes.append(c.codes_of(ids, root)); spec.append(c.spectra_of(ids, root))
        K += [int(r["K"]) for r in idx]; src += [name] * len(idx)
        if name == "dataset":
            ds_ids = ids
        else:
            pool_ids = ids
    codes, spec, K, src = np.concatenate(codes), np.concatenate(spec), np.array(K), np.array(src)
    ids = ds_ids + pool_ids
    cap = c.total_capacitance_nf(codes)
    out: dict = {"n_placements": int(len(K)), "n_dataset": int((src == "dataset").sum()), "n_pool": int((src == "pool").sum())}

    # 1. capacitance law at 1 MHz
    const = spec[:, 0] * 2 * np.pi * f[0] * 1e6 * cap * 1e-9
    out["capacitance_law"] = {"constant_median": float(np.median(const)), "constant_min": float(const.min()),
                              "constant_max": float(const.max()), "constant_p01": float(np.quantile(const, 0.01)),
                              "constant_p99": float(np.quantile(const, 0.99))}
    # 2. the T1 pairs, with the re-simulated spectra where the first slot is mounted
    pair = np.flatnonzero((K == 2) & ((codes == 1).sum(1) == 2))
    zp = spec[pair].copy()
    diffs = []
    for a, i in enumerate(pair):
        if ids[i] in resim:
            diffs.append(float(np.max(np.abs(resim[ids[i]] / zp[a] - 1))))
            zp[a] = resim[ids[i]]
    out["t1_pairs"] = {"n": int(len(pair)), "distinct": int(len({codes[i].tobytes() for i in pair})),
                       "n_resimulated": len(diffs), "resim_max_rel_diff_vs_stored": max(diffs) if diffs else None,
                       "resim_n_differ_above_1pct": int(sum(d > 0.01 for d in diffs))}
    # 3. three T1 decaps: 300 nF, about 1 % less than the 0.5-ohm target needs at the median constant, so the
    #    capacitance law alone does not exclude them, and only a sample of all such placements is simulated.
    #    The two requirements are set against each other: the impedance at 1 MHz and the worst-case ratio.
    from math import comb
    from scipy.stats import spearmanr
    xy, ic = c.slot_xy_mm()
    dist = np.hypot(xy[:, 0] - ic[0], xy[:, 1] - ic[1])
    tri = np.flatnonzero((K == 3) & ((codes == 1).sum(1) == 3))
    mean_d = lambda rows: np.array([dist[np.flatnonzero(codes[i] > 0)].mean() for i in rows])
    out["t1_triples"] = {"n_possible": comb(c.N_SLOTS, 3), "n_simulated": int(len(tri)),
                         "distinct": int(len({codes[i].tobytes() for i in tri})),
                         "rank_corr_1mhz_impedance_vs_distance_to_ic": float(spearmanr(spec[tri, 0], mean_d(tri)).correlation),
                         "pairs_rank_corr_1mhz_impedance_vs_distance_to_ic": float(spearmanr(zp[:, 0], mean_d(pair)).correlation),
                         "targets": {}}
    rule = {int(r["K"]): Z[i] for i, r in enumerate(man["spectra"]) if r["kind"] == "rule"}
    out["rule_of_thumb"] = {"closest_slots": man["closest_slots"], "distance_mm": man["distance_mm"], "targets": {}}
    out["targets"] = {}
    for t, zt in tg.items():
        rho = c.worst_ratio(spec, zt)
        ok1 = spec[:, 0] <= zt[0]
        need_nf = float(np.median(const)) / (2 * np.pi * f[0] * 1e6 * zt[0]) * 1e9
        rp = c.worst_ratio(zp, zt)
        worst_at = f[(zp / zt).argmax(1)]
        best = int(rp.argmin())
        rt, z1t = rho[tri], spec[tri, 0]
        b3 = int(rt.argmin())
        out["t1_triples"]["targets"][t] = {
            "n_pass": int((rt <= 1).sum()), "best_ratio": float(rt.min()), "best_worst_at_mhz": float(f[(spec[tri[b3]] / zt).argmax()]),
            "best_ratio_1mhz": float(z1t[b3] / zt[0]), "n_pass_1mhz": int((z1t <= zt[0]).sum()),
            "lowest_ratio_of_those_passing_1mhz": float(rt[z1t <= zt[0]].min()) if (z1t <= zt[0]).any() else None,
            "n_at_most_1pct_above_target_at_1mhz": int((z1t <= 1.01 * zt[0]).sum()),
            "lowest_ratio_of_those": float(rt[z1t <= 1.01 * zt[0]].min()) if (z1t <= 1.01 * zt[0]).any() else None,
            "rank_corr_1mhz_impedance_vs_ratio": float(spearmanr(z1t, rt).correlation),
            "dataset_and_pool_best_three_decaps_any_type": float(rho[K == 3].min())}
        res = {"target_1mhz_ohm": float(zt[0]), "capacitance_needed_nf": need_nf,
               "n_less_capacitance": int((cap < need_nf).sum()), "n_less_capacitance_passing_1mhz": int(((cap < need_nf) & ok1).sum()),
               "min_decaps_by_capacitance": int(np.ceil(need_nf / c.CAP_NF[1])),
               "smallest_capacitance_passing_1mhz_nf": float(cap[ok1].min()), "smallest_K_passing_1mhz": int(K[ok1].min()),
               "largest_capacitance_failing_1mhz_nf": float(cap[~ok1].max()),
               "smallest_K_with_passing_placement": int(K[rho <= 1].min()),
               "passing_share_dataset_pct": {int(k): c.share(rho[(src == "dataset") & (K == k)] <= 1) for k in range(1, 11)},
               "passing_n_dataset": {int(k): int((rho[(src == "dataset") & (K == k)] <= 1).sum()) for k in range(1, 11)},
               "best_ratio_dataset": {int(k): float(rho[(src == "dataset") & (K == k)].min()) for k in range(1, 11)},
               "best_ratio_all_simulated": {int(k): float(rho[K == k].min()) for k in range(1, 11)},
               "t1_pairs": {"n_pass": int((rp <= 1).sum()), "best_ratio": float(rp.min()),
                            "best_pair_slots": [int(s) + 1 for s in np.flatnonzero(codes[pair[best]] > 0)],
                            "best_pair_worst_at_mhz": float(worst_at[best]), "ratio_typical": c.typical(rp),
                            "worst_point_below_10mhz_pct": c.share(worst_at < 10), "ratio_1mhz": float(zp[best, 0] / zt[0])}}
        out["targets"][t] = res
        rr = {k: float(c.worst_ratio(z, zt)) for k, z in rule.items()}
        passing = [k for k, v in rr.items() if v <= 1]
        out["rule_of_thumb"]["targets"][t] = {"ratio": rr, "smallest_K_passing": min(passing) if passing else None}
    c.save_json(c.ev_out("ch4_budget_proof.json"), out)
    np.savez_compressed(c.ev_out("ch4_budget_proof.npz"), K=K, cap_nf=cap, z1=spec[:, 0], is_dataset=src == "dataset",
                        rho=np.stack([c.worst_ratio(spec, zt) for zt in tg.values()], 1), pair_spectra=zp,
                        pair_codes=codes[pair], rule_spectra=np.stack([rule[k] for k in sorted(rule)]),
                        rule_codes=np.array([r["codes"] for r in man["spectra"] if r["kind"] == "rule"]))
    print(json.dumps(out, indent=1))
    return out


def main() -> None:
    st = ecad.store(STORE)
    man_p = st / "manifest.json"
    man = json.loads(man_p.read_text()) if man_p.is_file() else prepare(st)
    ecad.simulate(st, man["spectra"], [])
    analyze(st, man)


if __name__ == "__main__":
    main()
