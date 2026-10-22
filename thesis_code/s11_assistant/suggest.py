"""Design assistant: score a placement against a target spectrum and, if it fails, find the nearest
placements that pass.

What: 1. predicts the spectrum of the placement with the placement-only encoder (231 points, 1-600 MHz);
    2. PASS when Z(f) <= (1 - m) x target(f) at every frequency of the checked band, with m the
       zero-false-pass margin measured for that target (SAFETY_MARGIN overrides it, e.g. 0);
    3. if it fails, searches nearest first. One change is
           RETYPE  another part in an occupied slot         (K unchanged)
           MOVE    a decap to a different empty slot        (K unchanged)
       (ADD and REMOVE exist but are off: the number of decaps stays the same.) Radius 1 is
       enumerated, then radius 2; radius 3 and more is sampled. Results are ranked by the number of
       changed decaps, then type changes before moves, then by headroom under the bar.
    The verdicts come from the surrogate, not from eCADSTAR; VALIDATE_ON_INDEPENDENT reports how often
    the surrogate's call matches the call from the simulated spectra of the 480 independent placements.
Usage: python -m thesis_code.s11_assistant.suggest
    Prints the report and writes OUTPUT_DIR/s11_assistant/suggestions.json.
Config keys: LAYOUT_SLOTS / LAYOUT_TYPES (or LAYOUT_CODES), TARGET_PATH, CHECK_BAND_MHZ, SAFETY_MARGIN,
    MIN_SUGGESTIONS, MAX_SUGGESTIONS, MAX_RADIUS, RADIUS3_BUDGET, ALLOW_ADD, ALLOW_REMOVE,
    VALIDATE_ON_INDEPENDENT, SEED.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from thesis_code.common import settings
from thesis_code.common.board import INPUTS
from thesis_code.common.frequency import freqs_mhz
from thesis_code.common.paths import need, out_dir
from thesis_code.s09_inference.placement_encoder import load, margin as measured_margin

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
# The placement to score. Slots are 1-based (as on the schematic); types are
# 1 = 100 nF / 222 pH / 8.9 mOhm, 2 = 47 nF / 154 pH / 21.4 mOhm, 3 = 22 nF / 142 pH / 25.2 mOhm.
LAYOUT_SLOTS: list[int] = [5, 19, 30, 49, 50]
LAYOUT_TYPES: list[int] = [3, 3, 3, 3, 3]
LAYOUT_CODES: list[int] | None = None      # or all 52 codes directly; overrides the above

TARGET_PATH = INPUTS / "target_impedance_0.8.npy"
CHECK_BAND_MHZ: tuple[float, float] = (1.0, 600.0)
SAFETY_MARGIN: float | None = None   # None = the measured zero-false-pass margin for TARGET_PATH

MIN_SUGGESTIONS = 1
MAX_SUGGESTIONS = 5
MAX_RADIUS = 3                  # 1 = single edit, 2 = two edits, ...
RADIUS3_BUDGET = 400_000        # radius >= 3 is sampled, not enumerated
ALLOW_ADD = False               # allow K + 1 (off: the number of decaps stays the same)
ALLOW_REMOVE = False            # allow K - 1
VALIDATE_ON_INDEPENDENT = True  # verdicts vs the simulated spectra of the independent placements
SEED = 42
# =============================================================================

N_SLOTS = 52
N_TYPES = 3
TYPE_NAME = {1: "100nF", 2: "47nF", 3: "22nF"}


def _codes_from_config() -> np.ndarray:
    if LAYOUT_CODES is not None:
        c = np.asarray(LAYOUT_CODES, dtype=np.int64).reshape(-1)
        if c.size != N_SLOTS:
            raise SystemExit(f"LAYOUT_CODES must have {N_SLOTS} entries, got {c.size}")
        return c
    if len(LAYOUT_SLOTS) != len(LAYOUT_TYPES):
        raise SystemExit("LAYOUT_SLOTS and LAYOUT_TYPES must be the same length")
    c = np.zeros(N_SLOTS, dtype=np.int64)
    for s, t in zip(LAYOUT_SLOTS, LAYOUT_TYPES):
        if not 1 <= s <= N_SLOTS:
            raise SystemExit(f"slot {s} out of range 1..{N_SLOTS}")
        c[s - 1] = t
    return c


class Judge:
    """PASS/FAIL from the placement-only encoder with a safety margin.

    A placement passes when it predicts Z <= (1 - margin) x target at every checked point. ``score``
    ranks passing placements: the worst ratio divided by the bar (lower = more headroom).
    """

    def __init__(self, members: list, tgt: np.ndarray, band: np.ndarray):
        self.members, self.tgt, self.band = members, tgt, band
        self.k_range = members[0][1].k_range

    def spectra(self, codes) -> dict:
        codes = np.asarray(codes, np.int64).reshape(-1, N_SLOTS)
        return {name: p.predict_ohm(codes).reshape(len(codes), -1) for name, p, _ in self.members}

    def assess(self, codes):
        """-> (passes [B] bool, score [B], {member: worst ratio [B]})."""
        ratios = {n: (z / np.maximum(self.tgt, 1e-12))[:, self.band].max(1) for n, z in self.spectra(codes).items()}
        passes = np.ones(len(next(iter(ratios.values()))), dtype=bool)
        score = np.zeros(len(passes))
        for name, _, mg in self.members:
            passes &= ratios[name] <= 1.0 - mg
            score = np.maximum(score, ratios[name] / (1.0 - mg))
        return passes, score, ratios

    def describe(self) -> str:
        return " AND ".join(f"{n} <= {1 - mg:.2f}x target" for n, _, mg in self.members)


def load_judge(tgt: np.ndarray, band: np.ndarray, *, margin: float | None = None,
               target_path: Path | None = None) -> Judge:
    """The placement-only encoder as judge; ``margin`` None = the measured margin of ``target_path``."""
    mg = float(measured_margin(Path(target_path or TARGET_PATH))) if margin is None else float(margin)
    return Judge([(f"vae:{settings.HEAD}", load(), mg)], tgt, band)


def describe(codes: np.ndarray) -> str:
    occ = np.flatnonzero(codes)
    return "  ".join(f"C{i+1}:{TYPE_NAME.get(int(codes[i]), codes[i])}" for i in occ) or "(empty)"


def verdict(z_ohm: np.ndarray, tgt: np.ndarray, band: np.ndarray, f: np.ndarray,
            margin: float = 0.0) -> dict:
    """PASS when |Z| stays under (1 - margin) x the mask everywhere in the checked band."""
    mg = float(margin)
    ratio = z_ohm[band] / np.maximum(tgt[band], 1e-12)
    i = int(np.argmax(ratio))
    return {"passes": bool(ratio.max() <= 1.0 - mg), "margin": mg,
            "worst_ratio": float(ratio.max()),
            "worst_mhz": float(f[band][i]),
            "worst_z": float(z_ohm[band][i]),
            "worst_target": float(tgt[band][i]),
            "headroom": float(1.0 - ratio.max())}


# --------------------------------------------------------------------------------------
# neighbour enumeration
# --------------------------------------------------------------------------------------
def _edits(codes: np.ndarray, n_types: int) -> list[np.ndarray]:
    """Every single-edit neighbour of ``codes`` (retype / move; add / remove when allowed)."""
    out: list[np.ndarray] = []
    occ = np.flatnonzero(codes)
    empty = np.flatnonzero(codes == 0)
    for s in occ:                                    # RETYPE
        for t in range(1, n_types + 1):
            if t == codes[s]:
                continue
            c = codes.copy(); c[s] = t; out.append(c)
    for s in occ:                                    # MOVE
        for d in empty:
            c = codes.copy(); c[d] = c[s]; c[s] = 0; out.append(c)
    if ALLOW_ADD:
        for d in empty:
            for t in range(1, n_types + 1):
                c = codes.copy(); c[d] = t; out.append(c)
    if ALLOW_REMOVE and len(occ) > 1:
        for s in occ:
            c = codes.copy(); c[s] = 0; out.append(c)
    return out


def neighbours_at(codes: np.ndarray, radius: int, n_types: int, rng) -> np.ndarray:
    """All placements exactly ``radius`` edits away (sampled once radius >= 3)."""
    seen = {tuple(codes.tolist())}
    frontier = [codes]
    for r in range(radius):
        nxt: dict[tuple, np.ndarray] = {}
        budget = RADIUS3_BUDGET if r >= 2 else None
        for base in frontier:
            cand = _edits(base, n_types)
            if budget is not None and len(cand) > 0:
                idx = rng.choice(len(cand), size=min(len(cand), 64), replace=False)
                cand = [cand[i] for i in idx]
            for c in cand:
                k = tuple(c.tolist())
                if k not in seen:
                    nxt[k] = c
            if budget is not None and len(nxt) >= budget:
                break
        seen.update(nxt.keys())
        frontier = list(nxt.values())
        if not frontier:
            break
    return np.array(frontier, dtype=np.int64) if frontier else np.zeros((0, N_SLOTS), np.int64)


def edit_summary(base: np.ndarray, cand: np.ndarray) -> str:
    """Plain-English description of how ``cand`` differs from ``base``."""
    parts = []
    b_occ, c_occ = set(np.flatnonzero(base).tolist()), set(np.flatnonzero(cand).tolist())
    for s in sorted(b_occ & c_occ):
        if base[s] != cand[s]:
            parts.append(f"C{s+1}: {TYPE_NAME.get(int(base[s]))} -> {TYPE_NAME.get(int(cand[s]))}")
    removed, added = sorted(b_occ - c_occ), sorted(c_occ - b_occ)
    while removed and added:
        s, d = removed.pop(0), added.pop(0)
        tag = (f"C{s+1} -> C{d+1}" if base[s] == cand[d]
               else f"C{s+1}({TYPE_NAME.get(int(base[s]))}) -> C{d+1}({TYPE_NAME.get(int(cand[d]))})")
        parts.append("move " + tag)
    for s in removed:
        parts.append(f"remove C{s+1}({TYPE_NAME.get(int(base[s]))})")
    for d in added:
        parts.append(f"add C{d+1}({TYPE_NAME.get(int(cand[d]))})")
    return "; ".join(parts) or "(no change)"


def n_edits(base: np.ndarray, cand: np.ndarray) -> int:
    b_occ, c_occ = set(np.flatnonzero(base).tolist()), set(np.flatnonzero(cand).tolist())
    retype = sum(1 for s in (b_occ & c_occ) if base[s] != cand[s])
    rm, ad = len(b_occ - c_occ), len(c_occ - b_occ)
    return retype + max(rm, ad)


def n_moves(base: np.ndarray, cand: np.ndarray) -> int:
    """Changed decaps that are not a type change in place (moved, added or removed)."""
    b_occ, c_occ = set(np.flatnonzero(base).tolist()), set(np.flatnonzero(cand).tolist())
    return max(len(b_occ - c_occ), len(c_occ - b_occ))


# --------------------------------------------------------------------------------------
def validate_on_independent(judge: Judge, tgt, band) -> dict:
    """How often the judge's PASS/FAIL call matches the call from the simulated spectra of the
    independent placements (never trained on). A false PASS is the costly direction."""
    ind = need(settings.INDEPENDENT_DIR, "INDEPENDENT_DIR")
    rows = json.loads((ind / "unseen_manifest.json").read_text())["rows"]
    codes = np.array([r["type_codes"] for r in rows], dtype=np.int64)
    real = np.load(ind / "pred" / "impedance_real_ohm.npy").reshape(len(codes), -1)
    vr = (real / np.maximum(tgt, 1e-12))[:, band].max(1) <= 1.0
    vp = judge.assess(codes)[0]
    return {"n": len(rows), "agreement": float((vr == vp).mean()), "real_pass_rate": float(vr.mean()),
            "pred_pass_rate": float(vp.mean()), "false_pass": int((vp & ~vr).sum()),
            "false_fail": int((~vp & vr).sum())}


def find_fixes(codes: np.ndarray, judge: Judge, n_types: int, *, max_radius: int | None = None,
               min_suggestions: int | None = None, seed: int | None = None, log=print) -> list[dict]:
    """Nearest placements the judge passes: all of radius 1, then radius 2, ... (radius >= 3 sampled),
    stopping at the first radius that yields ``min_suggestions``. Sorted by the number of changed
    decaps, then type changes before moves (fewest moved decaps), then by headroom under the bar.
    Placements outside the trained K range are skipped."""
    max_radius = MAX_RADIUS if max_radius is None else max_radius
    min_suggestions = MIN_SUGGESTIONS if min_suggestions is None else min_suggestions
    rng = np.random.default_rng(SEED if seed is None else seed)
    k_lo, k_hi = judge.k_range
    found: list[dict] = []
    for r in range(1, max_radius + 1):
        cand = neighbours_at(codes, r, n_types, rng)
        dropped = 0
        if cand.shape[0]:
            kc = (cand != 0).sum(1)
            inside = (kc >= k_lo) & (kc <= k_hi)
            dropped, cand = int((~inside).sum()), cand[inside]
        if cand.shape[0] == 0:
            continue
        passes, score, ratios = judge.assess(cand)
        ok = np.flatnonzero(passes)
        if log:
            log(f"  radius {r}: {cand.shape[0]:,} candidate placements, {len(ok):,} pass"
                + (f"  ({dropped:,} skipped: K outside trained {k_lo}..{k_hi})" if dropped else ""))
        for i in ok:
            found.append({"codes": cand[i], "score": float(score[i]), "edits": n_edits(codes, cand[i]),
                          "moves": n_moves(codes, cand[i]),
                          "radius": r, "ratios": {k: float(v[i]) for k, v in ratios.items()}})
        found = [x for x in found if x["edits"] <= r]
        if len(found) >= min_suggestions:
            break
    found.sort(key=lambda x: (x["edits"], x["moves"], x["score"]))
    return found


def main() -> None:
    f = freqs_mhz()
    tgt = np.load(TARGET_PATH).reshape(-1)
    band = np.flatnonzero((f >= CHECK_BAND_MHZ[0]) & (f <= CHECK_BAND_MHZ[1]))
    print("=" * 74)
    print("DECAP PLACEMENT ASSISTANT")
    print("=" * 74)
    print(f"  target      : {TARGET_PATH}   band {CHECK_BAND_MHZ[0]:.0f}-{CHECK_BAND_MHZ[1]:.0f} MHz "
          f"({len(band)} of {len(f)} points)")
    judge = load_judge(tgt, band, margin=SAFETY_MARGIN)
    print(f"  judge       : PASS requires {judge.describe()}")
    k_lo, k_hi = judge.k_range

    if VALIDATE_ON_INDEPENDENT:
        v = validate_on_independent(judge, tgt, band)
        print(f"\n  verdicts on {v['n']} independent placements vs their simulated spectra: agreement "
              f"{v['agreement']:.1%}, false PASS {v['false_pass']}, false FAIL {v['false_fail']}")

    codes = _codes_from_config()
    K = int((codes != 0).sum())
    if not k_lo <= K <= k_hi:
        raise SystemExit(f"  your placement has K={K}, outside the trained range {k_lo}..{k_hi}")
    passes, _, _ = judge.assess(codes)
    spectra = judge.spectra(codes)
    member_v = {name: verdict(spectra[name][0], tgt, band, f, margin=mg) for name, _, mg in judge.members}
    print("\n" + "-" * 74)
    print(f"  YOUR PLACEMENT (K={K}):  {describe(codes)}")
    print("-" * 74)
    print(f"  verdict     : {'PASS' if passes[0] else 'FAIL'}")
    for name, vd in member_v.items():
        print(f"    worst {vd['worst_mhz']:.0f} MHz -- {vd['worst_z']:.3f} ohm vs target {vd['worst_target']:.3f} ohm "
              f"({vd['worst_ratio']:.2f}x, bar {1 - vd['margin']:.2f}x)")
    out: dict = {"layout": codes.tolist(), "K": K, "judge": "vae", "vae_head": settings.HEAD,
                 "verdict": {"passes": bool(passes[0]), "members": member_v}, "suggestions": []}
    if passes[0]:
        print("\n  Nothing to fix.")
    else:
        found = find_fixes(codes, judge, N_TYPES)
        if not found:
            print(f"\n  No passing placement within {MAX_RADIUS} changed decaps.")
        for n, s_ in enumerate(found[:MAX_SUGGESTIONS], 1):
            print(f"  #{n}  {s_['edits']} edit(s): {edit_summary(codes, s_['codes'])}   "
                  f"({s_['score']:.2f} of the bar)")
            out["suggestions"].append({"codes": s_["codes"].tolist(), "edits": s_["edits"], "moves": s_["moves"],
                                       "edit_summary": edit_summary(codes, s_["codes"]),
                                       "ratios": s_["ratios"], "score": s_["score"]})
    dest = out_dir("s11_assistant") / "suggestions.json"
    dest.write_text(json.dumps(out, indent=2, default=float))
    print(f"  Wrote {dest}")


if __name__ == "__main__":
    main()
