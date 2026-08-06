"""Evaluate model predictions vs ECADSTAR labels at off-anchor MHz frequencies.

Run:
    python active_learning_pi/al/evaluate_off_anchor.py"""
from __future__ import annotations

from collections import defaultdict
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


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks for ties (1-based), matching scipy.stats.rankdata(method='average')."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    n = len(x)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def _corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3:
        return None
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    return _corr(_rankdata(x), _rankdata(y))


def evaluate_acquisition_rank_quality(
    manifest: list[dict[str, Any]],
    scored: list[dict[str, Any]],
    *,
    off_anchor_mhz: list[float],
    selected: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Check whether acquisition badness ranks true p99 error on labeled layouts.

    Uses acquisition-time ``scored`` rows (badness / acquisition_score) joined to
    ECAD labels. Positive Spearman/Pearson ⇒ higher badness ↔ higher true error.

    Also reports **per-MHz** Spearman and a sample-size-weighted stratified mean,
    because pooling bands with different error scales can flip the overall sign.
    ``ranking_ok`` is True if either overall Spearman > 0 **or** stratified Spearman > 0
    with lift ≥ 1 (when n is large enough per band).
    """
    label_report = evaluate_labels_vs_predictions(
        manifest, scored, off_anchor_mhz=off_anchor_mhz,
    )
    err_by_id = {int(r["candidate_id"]): float(r["p99_abs_err"]) for r in label_report.get("rows", [])}
    mhz_by_id = {int(r["candidate_id"]): float(r["mhz"]) for r in label_report.get("rows", [])}
    score_by_id: dict[int, float] = {}
    for p in scored:
        cid = int(p["candidate_id"])
        if cid not in err_by_id:
            continue
        bad = p.get("badness", p.get("acquisition_score", p.get("uncertainty")))
        if bad is None:
            continue
        score_by_id[cid] = float(bad)

    ids = sorted(score_by_id.keys())
    out: dict[str, Any] = {
        "available": False,
        "n": len(ids),
        "off_anchor_mhz": [float(m) for m in off_anchor_mhz],
        "score_field": "badness",
        "error_field": "p99_abs_err",
    }
    if len(ids) < 3:
        out["reason"] = "need ≥3 labeled off-anchor layouts with scores"
        return out

    bad = np.asarray([score_by_id[i] for i in ids], dtype=np.float64)
    err = np.asarray([err_by_id[i] for i in ids], dtype=np.float64)
    mhz_arr = np.asarray([mhz_by_id[i] for i in ids], dtype=np.float64)
    spearman = _spearman(bad, err)
    pearson = _corr(bad, err)

    order = np.argsort(-bad)  # high badness first
    n = len(ids)
    top_n = max(1, n // 2)
    bot_n = max(1, n // 2)
    top_mean = float(err[order[:top_n]].mean())
    bot_mean = float(err[order[-bot_n:]].mean())
    overall_mean = float(err.mean())

    selected_ids: set[int] = set()
    if selected:
        selected_ids = {int(r["candidate_id"]) for r in selected}
    labeled_selected = [i for i in ids if i in selected_ids]
    selected_mean = (
        float(np.mean([err_by_id[i] for i in labeled_selected]))
        if labeled_selected else None
    )
    k_sel = len(labeled_selected) if labeled_selected else min(len(ids), max(1, n // 4))
    topk_mean = float(err[order[:k_sel]].mean()) if k_sel else None

    # Per-MHz stratified ranking (avoids cross-band scale confounding).
    by_mhz: dict[float, list[int]] = defaultdict(list)
    for i, cid in enumerate(ids):
        by_mhz[float(mhz_arr[i])].append(i)
    per_mhz: list[dict[str, Any]] = []
    strat_num = 0.0
    strat_den = 0.0
    for m, idxs in sorted(by_mhz.items()):
        if len(idxs) < 5:
            per_mhz.append({"mhz": m, "n": len(idxs), "spearman": None, "note": "n<5"})
            continue
        bi = bad[idxs]
        ei = err[idxs]
        sp_m = _spearman(bi, ei)
        per_mhz.append({"mhz": m, "n": len(idxs), "spearman": sp_m})
        if sp_m is not None:
            strat_num += float(sp_m) * len(idxs)
            strat_den += float(len(idxs))
    strat_spearman = (strat_num / strat_den) if strat_den > 0 else None

    # MHz-zscored badness vs error (same idea as stratified, single pooled test).
    z_bad = np.zeros_like(bad)
    for m, idxs in by_mhz.items():
        bi = bad[idxs]
        std = float(bi.std())
        if std < 1e-12:
            z_bad[idxs] = 0.0
        else:
            z_bad[idxs] = (bi - float(bi.mean())) / std
    spearman_mhz_z = _spearman(z_bad, err)

    rows = [
        {
            "candidate_id": int(cid),
            "mhz": float(mhz_by_id[cid]),
            "badness": float(score_by_id[cid]),
            "p99_abs_err": float(err_by_id[cid]),
            "selected": cid in selected_ids,
        }
        for cid in ids
    ]

    lift = (top_mean / bot_mean) if bot_mean > 1e-12 else None
    # Directional OK if overall Spearman>0 with lift≥1, or a meaningfully positive
    # MHz-stratified / mhz-zscored Spearman (threshold avoids near-zero noise).
    ranking_ok = bool(
        (spearman is not None and spearman > 0.0 and (lift is None or float(lift) >= 1.0))
        or (strat_spearman is not None and float(strat_spearman) > 0.05)
        or (
            spearman_mhz_z is not None
            and float(spearman_mhz_z) > 0.05
            and lift is not None
            and float(lift) >= 1.0
        )
    )

    out.update({
        "available": True,
        "spearman_badness_vs_p99_err": spearman,
        "pearson_badness_vs_p99_err": pearson,
        "spearman_stratified_mhz": strat_spearman,
        "spearman_mhz_zscored_badness": spearman_mhz_z,
        "per_mhz": per_mhz,
        "mean_p99_err_all_labeled": overall_mean,
        "mean_p99_err_top_half_badness": top_mean,
        "mean_p99_err_bottom_half_badness": bot_mean,
        "top_vs_bottom_half_lift": lift,
        "n_selected_labeled": len(labeled_selected),
        "mean_p99_err_selected": selected_mean,
        "mean_p99_err_top_k_by_badness": topk_mean,
        "selected_vs_all_lift": (
            (selected_mean / overall_mean)
            if selected_mean is not None and overall_mean > 1e-12 else None
        ),
        "ranking_ok": ranking_ok,
        "rows": rows,
    })
    return out
