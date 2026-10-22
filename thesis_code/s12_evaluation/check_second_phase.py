"""Error of a frequency point after each step of the second training phase.

What: the placement-only encoder is first distilled from the trained encoders and then fine-tuned
    jointly with a copy of the spectrum decoder (the head, settings.HEAD). Both states predict the
    spectra of the test placements from the placement alone, and the error |Z_pred / Z_sim - 1| is
    taken at every frequency point. Reported with the measure of the rest of Chapter 4 (typical =
    median over all points, nine of ten = 90th percentile) and, for comparison with the training log
    of the head (spec_med_rel), the median over placements of each placement's median error. The two
    states are compared on the same placements with the same measure, which isolates what the joint
    fine-tuning adds.
Usage: python -m thesis_code.s12_evaluation.check_second_phase     (CONFIG-only; GPU recommended, about a minute)
Outputs: ch4_second_phase.json
"""
from __future__ import annotations

import numpy as np

from thesis_code.common import settings
from thesis_code.s12_evaluation import common as c

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
OUT = "ch4_second_phase.json"
# =============================================================================


def main() -> None:
    import torch
    from thesis_code.s09_inference import placement_encoder
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test = c.split_ids("test")
    codes, Z = c.codes_of(test), c.spectra_of(test)
    out = {"checkpoint": settings.CHECKPOINT, "n_test": len(test), "states": {}}
    for name, head in (("distilled", None), ("fine_tuned", settings.HEAD)):
        p = placement_encoder.load(device=dev, head=head)
        e = np.abs(p.predict_ohm(codes).reshape(len(codes), -1) / Z - 1)
        out["states"][name] = {"head": head, "point_typical_pct": round(100 * float(np.median(e)), 3),
                               "point_nine_of_ten_pct": round(100 * float(np.quantile(e, 0.9)), 3),
                               "median_of_placement_medians_pct": round(100 * float(np.median(np.median(e, 1))), 3)}
    c.save_json(c.ev_out(OUT), out)
    print(out)


if __name__ == "__main__":
    main()
