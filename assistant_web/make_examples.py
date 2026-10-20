"""Write the prepared examples of the web page (``assistant_web/examples.json``) from saved thesis results.

What: four placements, each a one-click demo step, with where it comes from and, where it exists, its
    eCADSTAR result (text only; the page shows no simulated curves):
      1. the thesis assistant's own example (thesis_code s11 suggest.py, LAYOUT_SLOTS / LAYOUT_TYPES);
      2. the rule of thumb: T1 in the RULE_K empty slots closest to the IC port;
      3. an independent placement of the unseen sweep (never in the dataset) that fails SWEEP_TARGET
         in eCADSTAR and that the assistant repairs with one change (assistant sweep, 2026-10-17);
      4. the placement the guided search returns for 0.5 ohm at its smallest budget (Ch4 walkthrough).
Usage: python -m assistant_web.make_examples
Config keys: EXP_DIR, RULE_K, SWEEP_TARGET, SWEEP_INDEX, SEARCH_TARGET, OUT.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from assistant_web.engine import change_list, change_text
from thesis_code.common import settings
from thesis_code.common.board import INPUTS
from thesis_code.s11_assistant import suggest as S

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
EXP_DIR = settings.REPO / "experiments/exp096_k1k10_balanced30k_ep700"
RULE_K = 3                       # the smallest budget of the 0.8 ohm target (thesis Ch4)
SWEEP_TARGET = "0.6"
SWEEP_INDEX = 138                # independent placement 138: fails 0.6 ohm, repaired by one change
SEARCH_TARGET = "0.5"
OUT = Path(__file__).resolve().parent / "examples.json"
# =============================================================================

N_SLOTS = 52


def _codes(slots: list[int], types: list[int]) -> list[int]:
    c = [0] * N_SLOTS
    for s, t in zip(slots, types):
        c[s - 1] = int(t)
    return c


def _rho(z: np.ndarray, target: str) -> float:
    return float((z / np.load(INPUTS / f"target_impedance_{target}.npy").reshape(-1)).max())


def main() -> None:
    ex = []
    c = _codes(S.LAYOUT_SLOTS, S.LAYOUT_TYPES)
    ex.append({"id": "thesis", "name": "Thesis example (0.8 Ω)", "target": "0.8", "codes": c,
               "note": "Example of the thesis assistant script: five T3."})

    j = json.loads((INPUTS / "slot_xy_mm.json").read_text())
    xy, ic = np.array(j["slots"]), np.array(j["ic_port"])
    near = np.argsort(np.hypot(*(xy - ic).T))[:RULE_K]
    c = _codes([int(i) + 1 for i in sorted(near)], [1] * RULE_K)
    ex.append({"id": "rule", "name": f"Rule of thumb, K = {RULE_K} (0.8 Ω)", "target": "0.8", "codes": c,
               "note": f"T1 in the {RULE_K} slots closest to the IC port."})

    ind = Path(settings.INDEPENDENT_DIR)
    rows = json.loads((ind / "unseen_manifest.json").read_text())["rows"]
    real = np.load(ind / "pred" / "impedance_real_ohm.npy").reshape(len(rows), -1)
    c = [int(x) for x in rows[SWEEP_INDEX]["type_codes"]]
    store = np.load(EXP_DIR / "assistant_sweep_ecad" / "spectra.npz")
    sims = {tuple(x.tolist()): z for x, z in zip(store["codes"], store["values"])}
    cases = json.loads((EXP_DIR / "assistant_sweep_ecad" / f"propose_{SWEEP_TARGET}.json").read_text())["cases"]
    case = next(x for x in cases if x["index"] == SWEEP_INDEX)
    first = case["modes"]["safe"]["proposals"][0]
    rho_fix = _rho(sims[tuple(first["codes"])], SWEEP_TARGET)
    ex.append({"id": "sweep", "name": f"Unseen placement #{SWEEP_INDEX} ({SWEEP_TARGET} Ω)", "target": SWEEP_TARGET,
               "codes": c,
               "note": f"Not in the dataset. eCADSTAR: ρ = {_rho(real[SWEEP_INDEX], SWEEP_TARGET):.2f}; after "
                       f"{change_text(change_list(np.array(c), np.array(first['codes'])))}: ρ = {rho_fix:.2f}."})

    srch = json.loads((EXP_DIR / "evals" / "ch4_search.json").read_text())["labels"]["s0"][SEARCH_TARGET]
    k = srch["smallest_budget_direct"]
    row = srch["cells"][str(k)]["rows"][0]
    c = [int(x) for x in row["codes"]]
    ex.append({"id": "search", "name": f"Search result, K = {k} ({SEARCH_TARGET} Ω)", "target": SEARCH_TARGET,
               "codes": c,
               "note": f"Guided search, smallest budget (thesis Ch4). eCADSTAR: ρ = {row['sim']:.2f}."})

    OUT.write_text(json.dumps(ex, indent=2, ensure_ascii=False))
    for e in ex:
        print(f"{e['id']:7s} {e['target']} ohm  {e['note']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
