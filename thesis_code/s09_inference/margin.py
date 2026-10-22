"""Zero-false-pass margin of the placement-only encoder on the test placements.

What: the encoder passes a placement when its predicted worst-case ratio
    rho = max_f Z_pred(f) / Z_target(f) is at most 1 - m. On the test placements (never used to train
    the VAE or the encoder) the margin is the smallest m in 0.00, 0.01, ..., 0.60 at which no placement
    that really fails (simulated rho > 1) is passed. It is measured per target spectrum; the value to
    use is the largest over the targets. The share of real passes it still keeps is its cost. The 480
    independent placements are a second check of the chosen margin (false passes there should be 0).
Usage: python -m thesis_code.s09_inference.margin
    Writes OUTPUT_DIR/s09_inference/<HEAD>_margin.json (the format placement_encoder.margin reads;
    copy it next to the head to use it).
Config keys: TARGETS, MARGINS.
"""
from __future__ import annotations

import csv
import json

import numpy as np

from thesis_code.common import settings
from thesis_code.common.board import INPUTS
from thesis_code.common.paths import need, out_dir
from thesis_code.s04_dataset.loader import _split_indices_by_design
from thesis_code.s09_inference.placement_encoder import load

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
TARGETS = {"default (0.8)": "target_impedance_0.8.npy", "0.6": "target_impedance_0.6.npy",
           "0.5": "target_impedance_0.5.npy"}          # files in common/inputs/
MARGINS = [round(0.01 * i, 2) for i in range(61)]       # up to 0.60
TRAIN_SPLIT, SPLIT_SEED = 0.9, 42
# =============================================================================


def main() -> dict:
    store = need(settings.DATA_DIR / "normalized", "DATA_DIR")
    with (store / "manifest.csv").open() as fh:
        designs = sorted({r["design_id"] for r in csv.DictReader(fh)})
    _, va_i = _split_indices_by_design(designs, TRAIN_SPLIT, SPLIT_SEED)
    held = [designs[i] for i in va_i]
    judge = load()
    codes = np.stack([np.load(store / "layouts" / d / "type.npy").astype(np.int64).reshape(-1) for d in held])
    real = np.exp(np.stack([np.load(store / "layouts" / d / "imp.npy").reshape(-1) for d in held])
                  * judge.log_std + judge.log_mean)
    pred = judge.predict_ohm(codes)
    print(f"{len(held):,} test placements, spectrum median |error| {np.median(np.abs(pred - real) / real):.2%}")

    ind = need(settings.INDEPENDENT_DIR, "INDEPENDENT_DIR")
    man = json.loads((ind / "unseen_manifest.json").read_text())["rows"]
    u_codes = np.array([r["type_codes"] for r in man], np.int64)
    u_real = np.load(ind / "pred" / "impedance_real_ohm.npy").reshape(-1, 231).astype(float)
    u_pred = judge.predict_ohm(u_codes)

    res: dict = {"checkpoint": settings.CHECKPOINT, "head": settings.HEAD, "n_heldout": len(held), "targets": {}}
    for name, fname in TARGETS.items():
        tgt = np.load(INPUTS / fname).reshape(-1)
        rp, rr = (pred / tgt).max(1), (real / tgt).max(1)
        up, ur = (u_pred / tgt).max(1), (u_real / tgt).max(1)
        rows = [{"margin": m, "false_pass": int(((rp <= 1 - m) & (rr > 1)).sum()),
                 "pass_kept": float(((rp <= 1 - m) & (rr <= 1)).sum() / max((rr <= 1).sum(), 1))} for m in MARGINS]
        mg = next((r["margin"] for r in rows if r["false_pass"] == 0), None)
        at = next(r for r in rows if r["margin"] == mg) if mg is not None else None
        res["targets"][name] = {
            "path": f"thesis_code/common/inputs/{fname}", "real_pass_heldout": int((rr <= 1).sum()),
            "real_fail_heldout": int((rr > 1).sum()), "margin": mg,
            "pass_kept_at_margin": at["pass_kept"] if at else None, "sweep": rows,
            "unseen_real_pass": int((ur <= 1).sum()),
            "unseen_false_pass_at_margin": int(((up <= 1 - mg) & (ur > 1)).sum()) if mg is not None else None,
            "unseen_pass_kept_at_margin": (float(((up <= 1 - mg) & (ur <= 1)).sum() / max((ur <= 1).sum(), 1))
                                           if mg is not None else None),
            "median_signed_worst_ratio_err": float(np.median(rp / rr - 1))}
        t = res["targets"][name]
        print(f"  target {name:13s}: margin {mg} (keeps {t['pass_kept_at_margin']}); independent placements: "
              f"{t['unseen_false_pass_at_margin']} false pass")
    ms = [t["margin"] for t in res["targets"].values()]
    res["margin_to_use"] = None if None in ms else max(ms)
    out = out_dir("s09_inference") / f"{settings.HEAD}_margin.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"margin to use (largest over the targets): {res['margin_to_use']} -> {out}")
    return res


if __name__ == "__main__":
    main()
