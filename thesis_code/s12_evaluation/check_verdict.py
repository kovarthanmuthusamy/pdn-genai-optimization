"""Where the pass/fail verdict of the surrogate goes wrong, and what sets the margin.

What: three questions about the verdict, answered from the spectra stored by check_accuracy (no
    model, no eCADSTAR):
  1. By limiting region. A placement fails where its simulated spectrum comes closest to (or exceeds)
     the target; the frequency of that worst point puts each placement into one of the four frequency
     regions. Per region: how many failing placements the surrogate lets pass (margin 0), and the
     error of the predicted worst-case ratio. Comparing the regions avoids pooling placements that
     fail for different physical reasons; the null is "the false-pass rate is the same in every region".
  2. The margin as an extreme value. The margin is the smallest one with no false pass on the test
     placements, so one placement sets it. Reported: the failing placements with the lowest predicted
     ratio, the margin with and without the worst one, and the share of passing placements kept at a
     range of margins. On the independent placements (not used to set the margin): false passes and
     passes kept at the margin of the test placements.
  3. Verdict flips by one decap (check_one_decap_ecad): of the simulated flips, how many the surrogate
     predicts (and in which direction), and of the predicted flips, how many are real. The reference
     is "never predict a flip".
Usage: python -m thesis_code.s12_evaluation.check_verdict     (CONFIG-only; CPU; after check_accuracy)
Outputs: ch4_verdict.json
"""
from __future__ import annotations

import numpy as np

from thesis_code.s12_evaluation import common as c

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
MARGIN_SWEEP = (0.0, 0.1, 0.2, 0.3, 0.4)
N_WORST = 5
# =============================================================================


def _margin(rp_fail: np.ndarray) -> float:
    """Smallest multiple of 0.01 at which no failing placement is predicted to pass."""
    return float(np.ceil((1 - rp_fail.min()) * 100 + 1e-9) / 100) if (rp_fail <= 1).any() else 0.0


def main() -> None:
    f, tg = c.freqs_mhz(), c.targets()
    a = dict(np.load(c.ev_in("ch4_accuracy.npz"), allow_pickle=True))
    from thesis_code.s09_inference.placement_encoder import margin
    res: dict = {"targets": {}}
    for t, zt in tg.items():
        out: dict = {}
        m_file = float(margin(c.TARGET_PATHS[t]))          # the measured margin of the head
        for name in ("test", "ind"):
            sim, pred = a[f"{name}_sim"], a[f"{name}_pred"]
            rs, rp = c.worst_ratio(sim, zt), c.worst_ratio(pred, zt)
            fw = f[(sim / zt).argmax(1)]
            fail, ok = rs > 1, rs <= 1
            reg = {}
            for r, (lo, hi) in c.REGIONS.items():
                q = (fw >= lo) & (fw < hi)
                reg[r] = {"n": int(q.sum()), "share_pct": c.share(q), "n_fail": int((q & fail).sum()), "n_pass": int((q & ok).sum()),
                          "false_pass_n": int((q & fail & (rp <= 1)).sum()),
                          "false_pass_pct": c.share(rp[q & fail] <= 1) if (q & fail).any() else None,
                          "false_fail_n": int((q & ok & (rp > 1)).sum()),
                          "false_fail_pct": c.share(rp[q & ok] > 1) if (q & ok).any() else None,
                          "ratio_error_typical_pct": c.typical(np.abs(rp[q] / rs[q] - 1) * 100) if q.any() else None,
                          "ratio_signed_typical_pct": c.typical((rp[q] / rs[q] - 1) * 100) if q.any() else None}
            blk = {"n": int(len(rs)), "n_fail": int(fail.sum()), "n_pass": int(ok.sum()), "by_limiting_region": reg,
                   "false_pass_n": int((fail & (rp <= 1)).sum()), "false_fail_n": int((ok & (rp > 1)).sum()),
                   "at_margin_of_test": {"margin": m_file, "false_pass_n": int((fail & (rp <= 1 - m_file)).sum()),
                                         "passes_kept_pct": c.share(rp[ok] <= 1 - m_file)},
                   "sweep": {f"{m:.2f}": {"false_pass_n": int((fail & (rp <= 1 - m)).sum()), "passes_kept_pct": c.share(rp[ok] <= 1 - m)}
                             for m in MARGIN_SWEEP}}
            if name == "test":
                order = np.flatnonzero(fail)[np.argsort(rp[fail])]
                m_all = _margin(rp[fail])
                m_wo = _margin(rp[order[1:]])
                blk["margin"] = {"from_rule": m_all, "in_margin_file": m_file,
                                 "worst": [{"predicted": float(rp[i]), "simulated": float(rs[i]), "K": int((a["test_codes"][i] > 0).sum()),
                                            "worst_point_mhz": float(fw[i])} for i in order[:N_WORST]],
                                 "without_the_worst_placement": m_wo,
                                 "passes_kept_pct": c.share(rp[ok] <= 1 - m_all),
                                 "passes_kept_without_the_worst_pct": c.share(rp[ok] <= 1 - m_wo),
                                 "n_fail_within_10pct_of_the_target": int((fail & (rs <= 1.1)).sum()),
                                 "false_pass_of_those_pct": c.share(rp[fail & (rs <= 1.1)] <= 1),
                                 "false_pass_of_fail_above_10pct_pct": c.share(rp[fail & (rs > 1.1)] <= 1)}
            out["test" if name == "test" else "independent"] = blk
        res["targets"][t] = out

    # ---- flips by one decap ---------------------------------------------------------------------
    od_p = c.ev_in("ch4_one_decap.npz")
    if od_p.is_file():
        o = np.load(od_p, allow_pickle=True)
        kind, base, Z = o["spec_kind"], o["spec_base"], o["Z"]
        nb = len(o["K"])
        zb = np.stack([Z[(kind == "base") & (base == b)][0] for b in range(nb)])
        res["one_decap_flips"] = {}
        for v in ("swap", "move"):
            zv = np.stack([Z[(kind == v) & (base == b)][0] for b in range(nb)])
            res["one_decap_flips"][v] = {}
            for t, zt in tg.items():
                sb, sv = c.worst_ratio(zb, zt) <= 1, c.worst_ratio(zv, zt) <= 1
                pb, pv = c.worst_ratio(o["pred_base"], zt) <= 1, c.worst_ratio(o[f"pred_{v}"], zt) <= 1
                flip, pflip = sb != sv, pb != pv
                same_dir = flip & pflip & (sv == pv)
                res["one_decap_flips"][v][t] = {
                    "n": nb, "flips": int(flip.sum()), "flips_to_pass": int((flip & sv).sum()), "flips_to_fail": int((flip & ~sv).sum()),
                    "flips_predicted": int(same_dir.sum()), "flips_predicted_pct": c.share(same_dir[flip]) if flip.any() else None,
                    "predicted_flips": int(pflip.sum()), "predicted_flips_real": int(same_dir.sum()),
                    "predicted_flips_real_pct": c.share(same_dir[pflip]) if pflip.any() else None,
                    "verdict_of_changed_placement_right_pct": c.share(sv == pv),
                    "verdict_of_changed_placement_right_if_never_flip_pct": c.share(sv == sb)}
    c.save_json(c.ev_out("ch4_verdict.json"), res)
    for t, v in res["targets"].items():
        mm = v["test"]["margin"]
        print(f"{t}: margin {mm['from_rule']} (file {mm['in_margin_file']}), without the worst {mm['without_the_worst_placement']}; kept "
              f"{mm['passes_kept_pct']:.1f} % / {mm['passes_kept_without_the_worst_pct']:.1f} %; worst {[(round(w['predicted'], 2), round(w['simulated'], 2), w['K'], w['worst_point_mhz']) for w in mm['worst']]}")
        print(f"   fail within 10 % of the target: {mm['n_fail_within_10pct_of_the_target']} of {v['test']['n_fail']}, let pass {mm['false_pass_of_those_pct']:.1f} %; "
              f"fail by more: let pass {mm['false_pass_of_fail_above_10pct_pct']:.1f} %")
        for s in ("test", "independent"):
            b = v[s]
            print(f"   {s}: n {b['n']} fail {b['n_fail']} pass {b['n_pass']}; false pass {b['false_pass_n']}, false fail {b['false_fail_n']}; at margin "
                  f"{b['at_margin_of_test']['margin']}: false pass {b['at_margin_of_test']['false_pass_n']}, kept {b['at_margin_of_test']['passes_kept_pct']:.1f} %; "
                  f"sweep {[(k, x['false_pass_n'], round(x['passes_kept_pct'])) for k, x in b['sweep'].items()]}")
            for r, x in b["by_limiting_region"].items():
                print(f"      limited at {r:>7} MHz: n {x['n']:>4} ({x['share_pct']:.1f} %), fail {x['n_fail']:>4}, let pass {x['false_pass_n']:>3} "
                      f"({x['false_pass_pct'] if x['false_pass_pct'] is None else round(x['false_pass_pct'], 1)} %), false fail {x['false_fail_n']:>3} "
                      f"({x['false_fail_pct'] if x['false_fail_pct'] is None else round(x['false_fail_pct'], 1)} %), ratio error typical "
                      f"{x['ratio_error_typical_pct'] if x['ratio_error_typical_pct'] is None else round(x['ratio_error_typical_pct'], 1)} % "
                      f"(signed {x['ratio_signed_typical_pct'] if x['ratio_signed_typical_pct'] is None else round(x['ratio_signed_typical_pct'], 1)})")
    for v, d in res.get("one_decap_flips", {}).items():
        for t, x in d.items():
            print(f"one decap {v} {t}: flips {x['flips']} of {x['n']} (to pass {x['flips_to_pass']}, to fail {x['flips_to_fail']}); predicted {x['flips_predicted']} "
                  f"({x['flips_predicted_pct']:.0f} %); predicted flips {x['predicted_flips']}, real {x['predicted_flips_real']}; verdict of the changed placement right "
                  f"{x['verdict_of_changed_placement_right_pct']:.1f} % (never flip: {x['verdict_of_changed_placement_right_if_never_flip_pct']:.1f} %)")


if __name__ == "__main__":
    main()
