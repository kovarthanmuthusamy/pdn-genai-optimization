"""Beyond the training range: placements with 11 and 12 decaps, against eCADSTAR.

What: the surrogate was trained on budgets K = 1..10. N_PER_K placements each with K = 11 and K = 12
    are drawn with the recipe of the 480 independent placements (random slots, every type at least
    once, the remaining types at random) and simulated: spectrum and maps at the same eleven
    frequencies. The surrogate predicts them from the placement alone with the budget input held at
    10, the largest trained value ("clamped"). Compared with the accuracy on the independent
    placements with K = 9 and 10 (same recipe, same frequencies, same measures). Null: no loss of
    accuracy outside the range. The placement-only encoder refuses budgets outside the range; this
    check shows what that refusal protects against.
Usage: python -m thesis_code.s12_evaluation.check_beyond_range_ecad     (CONFIG-only; resumable;
       about 16 min of eCADSTAR when the store is new)
Outputs: store beyond_range_ecad/ (when new); ch4_beyond_range.json (+ .npz)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from thesis_code.s12_evaluation import common as c
from thesis_code.s12_evaluation import ecad

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
K_VALUES = (11, 12)
N_PER_K = 60
MAP_MHZ = [200.0, 3.0, 24.0, 58.0, 132.0, 234.0, 275.0, 407.0, 456.0, 514.0, 582.0]   # as the independent placements
SEED = 20261011
STORE = "beyond_range_ecad"
# =============================================================================


def prepare(st: Path) -> dict:
    rng = np.random.default_rng(SEED)
    spectra, maps = [], []
    for k in K_VALUES:
        for _ in range(N_PER_K):
            cd = np.zeros(c.N_SLOTS, dtype=int)
            types = np.concatenate([np.arange(1, 4), rng.integers(1, 4, size=k - 3)])
            rng.shuffle(types)
            cd[rng.choice(c.N_SLOTS, size=k, replace=False)] = types
            p = len(spectra)
            spectra.append({"placement": p, "K": int(k), "codes": cd.tolist()})
            maps += [{"placement": p, "K": int(k), "codes": cd.tolist(), "map_mhz": m} for m in MAP_MHZ]
    man = {"seed": SEED, "spectra": ecad.assign(spectra), "maps": ecad.assign(maps)}
    c.save_json(ecad.writable(st) / "manifest.json", man)
    print(f"{len(spectra)} spectra, {len(maps)} maps to simulate", flush=True)
    return man


def analyze(st: Path, man: dict) -> dict:
    mask, f, tg = c.board_mask(), c.freqs_mhz(), c.targets()
    Z, H = ecad.read_spectra(st, man["spectra"]), ecad.read_maps(st, man["maps"])
    cs = np.array([r["codes"] for r in man["spectra"]]); ks = np.array([r["K"] for r in man["spectra"]])
    cm = np.array([r["codes"] for r in man["maps"]]); km = np.array([r["K"] for r in man["maps"]])
    fm = np.array([r["map_mhz"] for r in man["maps"]])
    sur = c.Surrogate()
    out: dict = {"n_per_K": N_PER_K, "map_mhz": MAP_MHZ, "modes": {}}
    arrays = {"Z": Z, "H": H, "K": ks, "map_K": km, "map_mhz": fm, "codes": cs}
    # The budget input of the surrogate is a table with one entry per trained budget: a budget of 11 or 12
    # has no entry, so these placements are evaluated with the budget input held at 10 ("clamped").
    mode = "clamped"
    ps, pm = sur.spectra(cs, np.minimum(ks, c.K_RANGE[1])), sur.maps(cm, fm, np.minimum(km, c.K_RANGE[1]))
    rows = np.array([c.map_measures(pm[i], H[i], mask) for i in range(len(H))])
    res = {}
    for k in K_VALUES:
        res[int(k)] = {"spectrum": c.spectrum_summary(ps[ks == k], Z[ks == k], f, tg[c.PEAK_TARGET]),
                       "maps": c.map_summary(rows[km == k]),
                       "maps_other_than_200": c.map_summary(rows[(km == k) & (fm != c.F0_MHZ)]),
                       "verdict_agreement_pct": {t: c.share((c.worst_ratio(ps[ks == k], zt) <= 1) == (c.worst_ratio(Z[ks == k], zt) <= 1))
                                                 for t, zt in tg.items()},
                       "passing_share_simulated_pct": {t: c.share(c.worst_ratio(Z[ks == k], zt) <= 1) for t, zt in tg.items()}}
    out["modes"][mode] = res
    arrays.update({f"pred_{mode}": ps, f"maps_{mode}": rows})
    c.save_json(c.ev_out("ch4_beyond_range.json"), out)
    np.savez_compressed(c.ev_out("ch4_beyond_range.npz"), **arrays)
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
