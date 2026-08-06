"""Acquisition functions — select worst/uncertain candidates for ECADSTAR simulation.

Run:
    python active_learning_pi/al/acquisition.py"""
from __future__ import annotations

from typing import Any

import numpy as np


def _hamming(a: list[int], b: list[int]) -> int:
    return sum(int(x != y) for x, y in zip(a, b))


def add_badness_scores(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    badness = how much we need ECADStar labels (higher = worse / more uncertain).
    quality_score = -badness (lower = worse candidate).
    """
    for r in scored:
        bad = float(r.get("acquisition_score", r.get("uncertainty", 0.0)))
        r["badness"] = bad
        r["quality_score"] = -bad
    return scored


def select_worst_for_simulation(
    scored: list[dict[str, Any]],
    *,
    simulate_batch_size: int,
    min_uncertainty: float | None = None,
    min_uncertainty_percentile: float = 0.0,
    stratify_by_k: bool = False,
    mhz_quotas: dict[float, int] | None = None,
) -> list[dict[str, Any]]:
    """
    Pick only the worst candidates (highest uncertainty) for a single ECADStar batch.
    Skips cheap random/diversity picks — simulation budget goes to bad cases only.

    With ``stratify_by_k``, picks worst layouts per K bucket (e.g. 48 batch / 8 K → 6 each).
    With ``mhz_quotas``, picks worst per MHz bin (e.g. 2 each at 90/265/435/510 MHz).
    """
    if not scored:
        return []
    scored = add_badness_scores(list(scored))
    n = min(simulate_batch_size, len(scored))

    if mhz_quotas:
        return _select_worst_stratified_by_mhz(
            scored,
            mhz_quotas=mhz_quotas,
            n=n,
            min_uncertainty=min_uncertainty,
            min_uncertainty_percentile=min_uncertainty_percentile,
        )

    if stratify_by_k:
        return _select_worst_stratified_by_k(
            scored,
            n=n,
            min_uncertainty=min_uncertainty,
            min_uncertainty_percentile=min_uncertainty_percentile,
        )

    ranked = sorted(scored, key=lambda r: r["badness"], reverse=True)

    if min_uncertainty_percentile > 0:
        vals = [r["badness"] for r in scored]
        thresh = float(np.percentile(vals, min_uncertainty_percentile))
        ranked = [r for r in ranked if r["badness"] >= thresh]

    if min_uncertainty is not None:
        ranked = [r for r in ranked if r["badness"] >= min_uncertainty]

    selected = ranked[:n]
    for r in selected:
        r["selected_reason"] = "worst_uncertainty"
    return selected


def _select_worst_stratified_by_k(
    scored: list[dict[str, Any]],
    *,
    n: int,
    min_uncertainty: float | None,
    min_uncertainty_percentile: float,
) -> list[dict[str, Any]]:
    from collections import defaultdict

    pools: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in scored:
        pools[int(r["k"])].append(r)

    if min_uncertainty_percentile > 0:
        vals = [r["badness"] for r in scored]
        thresh = float(np.percentile(vals, min_uncertainty_percentile))
        for k in list(pools):
            pools[k] = [r for r in pools[k] if r["badness"] >= thresh]

    if min_uncertainty is not None:
        for k in list(pools):
            pools[k] = [r for r in pools[k] if r["badness"] >= min_uncertainty]

    k_sorted = sorted(pools)
    if not k_sorted:
        return []

    per_k = max(1, n // len(k_sorted))
    selected: list[dict[str, Any]] = []
    used: set[int] = set()

    for k in k_sorted:
        ranked = sorted(pools[k], key=lambda r: r["badness"], reverse=True)
        for r in ranked[:per_k]:
            cid = int(r["candidate_id"])
            if cid not in used:
                selected.append(r)
                used.add(cid)
            if len(selected) >= n:
                break
        if len(selected) >= n:
            break

    if len(selected) < n:
        ranked_all = sorted(scored, key=lambda r: r["badness"], reverse=True)
        for r in ranked_all:
            cid = int(r["candidate_id"])
            if cid in used:
                continue
            selected.append(r)
            used.add(cid)
            if len(selected) >= n:
                break

    for r in selected:
        r["selected_reason"] = "worst_uncertainty_stratified_k"
    return selected[:n]


def _mhz_match(mhz: float) -> float:
    return round(float(mhz), 6)


def _select_worst_stratified_by_mhz(
    scored: list[dict[str, Any]],
    *,
    mhz_quotas: dict[float, int],
    n: int,
    min_uncertainty: float | None,
    min_uncertainty_percentile: float,
) -> list[dict[str, Any]]:
    from collections import defaultdict

    quota = {_mhz_match(m): int(c) for m, c in mhz_quotas.items()}
    pools: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for r in scored:
        pools[_mhz_match(float(r["mhz"]))].append(r)

    if min_uncertainty_percentile > 0:
        vals = [r["badness"] for r in scored]
        thresh = float(np.percentile(vals, min_uncertainty_percentile))
        for m in list(pools):
            pools[m] = [r for r in pools[m] if r["badness"] >= thresh]

    if min_uncertainty is not None:
        for m in list(pools):
            pools[m] = [r for r in pools[m] if r["badness"] >= min_uncertainty]

    selected: list[dict[str, Any]] = []
    used: set[int] = set()

    for mhz in sorted(quota):
        take = quota[mhz]
        ranked = sorted(pools.get(mhz, []), key=lambda r: r["badness"], reverse=True)
        for r in ranked[:take]:
            cid = int(r["candidate_id"])
            if cid not in used:
                selected.append(r)
                used.add(cid)

    if len(selected) < n:
        ranked_all = sorted(scored, key=lambda r: r["badness"], reverse=True)
        for r in ranked_all:
            cid = int(r["candidate_id"])
            if cid in used:
                continue
            selected.append(r)
            used.add(cid)
            if len(selected) >= n:
                break

    for r in selected:
        r["selected_reason"] = "worst_uncertainty_stratified_mhz"
    return selected[:n]


def select_batch(
    scored: list[dict[str, Any]],
    *,
    batch_size: int,
    top_n_fraction: float = 0.6,
    random_fraction: float = 0.2,
    diversity_fraction: float = 0.2,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Legacy mixed acquisition (optional; not used for ECADStar batch by default)."""
    if not scored:
        return []
    batch_size = min(batch_size, len(scored))
    rng = np.random.default_rng(seed)

    n_top = max(1, int(round(batch_size * top_n_fraction)))
    n_rand = max(0, int(round(batch_size * random_fraction)))
    n_div = max(0, batch_size - n_top - n_rand)

    ranked = sorted(scored, key=lambda r: r["acquisition_score"], reverse=True)
    selected: list[dict[str, Any]] = []
    used_ids: set[int] = set()

    for r in ranked:
        if len(selected) >= n_top:
            break
        cid = r["candidate_id"]
        if cid not in used_ids:
            selected.append(r)
            used_ids.add(cid)

    pool = [r for r in ranked if r["candidate_id"] not in used_ids]
    if n_rand > 0 and pool:
        picks = rng.choice(len(pool), size=min(n_rand, len(pool)), replace=False)
        for i in picks:
            r = pool[int(i)]
            selected.append(r)
            used_ids.add(r["candidate_id"])

    pool = [r for r in ranked if r["candidate_id"] not in used_ids]
    for _ in range(n_div):
        if not pool:
            break
        if not selected:
            selected.append(pool[0])
            used_ids.add(pool[0]["candidate_id"])
            pool = pool[1:]
            continue
        best_idx = 0
        best_min_dist = -1
        for i, r in enumerate(pool):
            occ = r["occupancy"]
            d = min(_hamming(occ, s["occupancy"]) for s in selected)
            if d > best_min_dist:
                best_min_dist = d
                best_idx = i
        r = pool.pop(best_idx)
        selected.append(r)
        used_ids.add(r["candidate_id"])

    while len(selected) < batch_size:
        for r in ranked:
            if r["candidate_id"] not in used_ids:
                selected.append(r)
                used_ids.add(r["candidate_id"])
                if len(selected) >= batch_size:
                    break
        break

    return selected[:batch_size]
