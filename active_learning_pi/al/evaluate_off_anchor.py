"""Evaluate model predictions vs ECADSTAR labels at off-anchor MHz frequencies.

Run:
    python active_learning_pi/al/evaluate_off_anchor.py"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from active_learning_pi.al.robust_stats import robust_peak_stats


def evaluate_labels_vs_predictions(
    manifest: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    off_anchor_mhz: list[float],
) -> dict[str, Any]:
    """
    Compare ingested real heatmaps to model predictions (pred_* fields) at off-anchor MHz.
    """
    pred_by_id = {p["candidate_id"]: p for p in predictions}
    off_set = {float(m) for m in off_anchor_mhz}
    rows = []

    for m in manifest:
        if not m.get("ingest_ok"):
            continue
        mhz = float(m["mhz"])
        if mhz not in off_set:
            continue
        cid = m["candidate_id"]
        pred = pred_by_id.get(cid)
        if pred is None:
            continue
        hm_path = Path(m["sample_dir"]) / "heatmap.npy"
        if not hm_path.is_file():
            continue
        hm = np.load(hm_path)
        real_stats = robust_peak_stats(hm)
        rows.append({
            "candidate_id": cid,
            "mhz": mhz,
            "real_p99": real_stats["p99"],
            "pred_p99_mean": pred.get("pred_p99_mean"),
            "p99_abs_err": abs(real_stats["p99"] - float(pred.get("pred_p99_mean", 0))),
        })

    if not rows:
        return {"n": 0, "rows": []}

    errs = [r["p99_abs_err"] for r in rows]
    return {
        "n": len(rows),
        "p99_mae": float(np.mean(errs)),
        "p99_median_ae": float(np.median(errs)),
        "rows": rows,
    }
