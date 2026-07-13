#!/usr/bin/env python3
"""Sample new decap layout combinations for PI simulation (inverse-K, exclude existing).

Reads the existing ``all_combinations.csv``, then writes ``combinations.csv`` with
``TOTAL_N`` new 52-d binary rows **not** present in the old file.

- **K=2:** every remaining unique pair layout (all C(52,2) not already in the old CSV).
- **K=3..50:** inverse-exponential counts for the rest of ``TOTAL_N``.

Run:
    python pipelines/heatmaps/sample_new_combinations.py

Agent notes:
    - Old layouts: ``EXISTING_CSV`` (default ``data/heatmaps/all_combinations.csv``)
    - Output: ``OUTPUT_CSV`` (default ``data/heatmaps/combinations.csv``)
    - Optional ``CANDIDATE_POOL_CSV``: larger CSV to draw from instead of random gen
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO_BOOT = Path(__file__).resolve().parents[2]
if str(_REPO_BOOT) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOT))

from repo_paths import REPO_ROOT as _ROOT, setup_path

setup_path()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# =============================================================================
# CONFIGURATION — edit before running
# =============================================================================

EXISTING_CSV = _ROOT / "data" / "heatmaps" / "all_combinations.csv"
OUTPUT_CSV = _ROOT / "data" / "heatmaps" / "combinations.csv"
REPORT_JSON = _ROOT / "data" / "heatmaps" / "combinations_sample_report.json"

# Optional: full candidate pool (52-col CSV). None = generate random K-hot layouts.
CANDIDATE_POOL_CSV: Path | None = None

TOTAL_N = 10_000
SEED = 42
N_DECAPS = 52

# Inverse-K schedule (same shape as subsample_inverse_k.py / original all_combinations.csv)
N_REF = 1000.0
TAU = 12.0
TAU_LOW = 35.0
K_LOW_MAX = 20
K_ANCHOR = 2
N_MIN = 150
N_MAX = 1000
K_MIN = 2
K_INVERSE_MIN = 3  # inverse-K loop starts here (K=2 is always "all remaining")
K_MAX = 50
EDGE_K = frozenset({0, 1, 51, 52})
INCLUDE_ALL_K2 = True

MAX_GEN_ATTEMPTS_PER_SLOT = 500_000

# =============================================================================


def row_tuple(values: list[int] | np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in values)


def row_k(values: list[int] | np.ndarray) -> int:
    return int(sum(int(x) for x in values))


def load_csv_rows(path: Path) -> list[tuple[int, ...]]:
    rows: list[tuple[int, ...]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != N_DECAPS:
                raise ValueError(f"{path}: expected {N_DECAPS} columns, got {len(parts)}")
            rows.append(row_tuple(int(x) for x in parts))
    return rows


def comb_capacity(k: int) -> int:
    """Number of distinct K-hot layouts on N_DECAPS sites."""
    if k < 0 or k > N_DECAPS:
        return 0
    return math.comb(N_DECAPS, k)


def existing_counts_by_k(existing: set[tuple[int, ...]]) -> Counter[int]:
    return Counter(row_k(r) for r in existing)


def cap_targets_by_availability(
    targets: dict[int, int],
    existing_by_k: Counter[int],
) -> tuple[dict[int, int], dict[int, int]]:
    """Cap per-K targets by C(52,K) minus already-used layouts; redistribute deficit."""
    capped = dict(targets)
    available = {
        k: max(0, comb_capacity(k) - int(existing_by_k.get(k, 0)))
        for k in capped
    }

    for k in sorted(capped):
        capped[k] = min(capped[k], available[k])

    shortfall = sum(targets.values()) - sum(capped.values())
    if shortfall <= 0:
        return capped, available

    # Redistribute to K with spare capacity (prefer larger slack, then lower K)
    while shortfall > 0:
        candidates = [
            k
            for k in capped
            if capped[k] < available[k]
        ]
        if not candidates:
            raise RuntimeError(
                f"Cannot allocate {sum(targets.values()):,} unique layouts: "
                f"shortfall={shortfall:,} but no K bucket has spare capacity"
            )
        candidates.sort(key=lambda k: (available[k] - capped[k], -k), reverse=True)
        k = candidates[0]
        capped[k] += 1
        shortfall -= 1

    return capped, available


def inverse_k_targets(
    total_n: int,
    *,
    k_min: int = K_INVERSE_MIN,
    k_max: int = K_MAX,
    k_anchor: int | None = None,
) -> dict[int, int]:
    """Per-K counts scaled to ``total_n`` (inverse-exponential over k_min..k_max)."""
    if total_n <= 0:
        return {}
    anchor = k_anchor if k_anchor is not None else k_min
    k_values = [k for k in range(k_min, k_max + 1) if k not in EDGE_K]
    if not k_values:
        raise ValueError(f"no K values in range {k_min}..{k_max}")

    raw: dict[int, float] = {}
    for k in k_values:
        tau_eff = TAU_LOW if k <= K_LOW_MAX else TAU
        raw_n = N_REF * math.exp(-(k - anchor) / tau_eff)
        capped = max(float(N_MIN), min(float(N_MAX), raw_n))
        raw[k] = capped

    weight_sum = sum(raw.values())
    targets = {k: max(1, int(round(total_n * raw[k] / weight_sum))) for k in k_values}

    delta = total_n - sum(targets.values())
    if delta > 0:
        for k in sorted(k_values, key=lambda x: raw[x], reverse=True):
            if delta <= 0:
                break
            targets[k] += 1
            delta -= 1
    elif delta < 0:
        for k in sorted(k_values, key=lambda x: raw[x]):
            if delta >= 0:
                break
            if targets[k] > 1:
                targets[k] -= 1
                delta += 1

    if sum(targets.values()) != total_n:
        raise RuntimeError(f"allocation failed: sum={sum(targets.values())} != {total_n}")
    return targets


def all_remaining_k2_layouts(exclude: set[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """Enumerate every K=2 layout not in ``exclude`` (C(52,2) pairs)."""
    rows: list[tuple[int, ...]] = []
    for i in range(N_DECAPS):
        for j in range(i + 1, N_DECAPS):
            row = [0] * N_DECAPS
            row[i] = 1
            row[j] = 1
            t = tuple(row)
            if t not in exclude:
                rows.append(t)
    return rows


def build_targets(existing_by_k: Counter[int], total_n: int) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    """K=2 = all available; K=3..50 = inverse-K on the remainder of ``total_n``."""
    available = {
        k: max(0, comb_capacity(k) - int(existing_by_k.get(k, 0)))
        for k in range(K_MIN, K_MAX + 1)
        if k not in EDGE_K
    }

    targets: dict[int, int] = {}
    raw_inverse: dict[int, int] = {}

    k2_n = 0
    if INCLUDE_ALL_K2:
        k2_n = available.get(2, 0)
        if k2_n > total_n:
            raise RuntimeError(
                f"all K=2 layouts ({k2_n:,}) exceed TOTAL_N ({total_n:,}); raise TOTAL_N"
            )
        targets[2] = k2_n

    remainder = total_n - k2_n
    if remainder > 0:
        raw_inverse = inverse_k_targets(remainder, k_min=K_INVERSE_MIN, k_anchor=K_INVERSE_MIN)
        capped, _ = cap_targets_by_availability(raw_inverse, existing_by_k)
        targets.update(capped)

    if sum(targets.values()) != total_n:
        raise RuntimeError(
            f"target build failed: sum={sum(targets.values())} != {total_n} "
            f"(K2={k2_n}, inverse remainder={remainder})"
        )
    return targets, available, raw_inverse


def random_k_hot(k: int, rng: random.Random) -> tuple[int, ...]:
    if k <= 0:
        return tuple(0 for _ in range(N_DECAPS))
    if k > N_DECAPS:
        raise ValueError(f"K={k} > {N_DECAPS}")
    idx = rng.sample(range(N_DECAPS), k)
    row = [0] * N_DECAPS
    for i in idx:
        row[i] = 1
    return tuple(row)


def sample_from_pool(
    pool_by_k: dict[int, list[tuple[int, ...]]],
    targets: dict[int, int],
    *,
    exclude: set[tuple[int, ...]],
    rng: random.Random,
) -> list[tuple[int, ...]]:
    selected: list[tuple[int, ...]] = []
    picked: set[tuple[int, ...]] = set()

    for k in sorted(targets):
        need = targets[k]
        if k == 2 and INCLUDE_ALL_K2:
            k2_rows = all_remaining_k2_layouts(exclude | picked)
            if len(k2_rows) != need:
                raise RuntimeError(f"K=2: expected {need} exhaustive layouts, got {len(k2_rows)}")
            selected.extend(k2_rows)
            picked.update(k2_rows)
            continue

        candidates = [r for r in pool_by_k.get(k, []) if r not in exclude and r not in picked]
        if len(candidates) < need:
            raise RuntimeError(
                f"K={k}: need {need} new layouts but only {len(candidates)} available in pool "
                f"(after excluding existing)"
            )
        rng.shuffle(candidates)
        take = candidates[:need]
        selected.extend(take)
        picked.update(take)

    return selected


def generate_random(
    targets: dict[int, int],
    *,
    exclude: set[tuple[int, ...]],
    rng: random.Random,
) -> list[tuple[int, ...]]:
    selected: list[tuple[int, ...]] = []
    picked: set[tuple[int, ...]] = set()

    for k in sorted(targets):
        need = targets[k]
        if k == 2 and INCLUDE_ALL_K2:
            k2_rows = all_remaining_k2_layouts(exclude | picked)
            if len(k2_rows) != need:
                raise RuntimeError(f"K=2: expected {need} exhaustive layouts, got {len(k2_rows)}")
            selected.extend(k2_rows)
            picked.update(k2_rows)
            continue

        got = 0
        attempts = 0
        while got < need:
            attempts += 1
            if attempts > MAX_GEN_ATTEMPTS_PER_SLOT:
                raise RuntimeError(
                    f"K={k}: could not generate {need} unique layouts after "
                    f"{MAX_GEN_ATTEMPTS_PER_SLOT} attempts (got {got})"
                )
            row = random_k_hot(k, rng)
            if row in exclude or row in picked:
                continue
            picked.add(row)
            selected.append(row)
            got += 1

    return selected


def write_csv(path: Path, rows: list[tuple[int, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for row in rows:
            w.writerow(list(row))


def main() -> None:
    if not EXISTING_CSV.is_file():
        raise SystemExit(f"Missing existing CSV: {EXISTING_CSV}")

    existing_rows = load_csv_rows(EXISTING_CSV)
    existing_set = set(existing_rows)
    existing_by_k = existing_counts_by_k(existing_set)
    print(f"Existing layouts: {len(existing_rows):,} rows ({len(existing_set):,} unique)")

    targets, available, raw_inverse = build_targets(existing_by_k, TOTAL_N)

    k2_n = targets.get(2, 0)
    print(f"K=2: all remaining pairs → {k2_n:,} layouts")
    print(
        f"K={K_INVERSE_MIN}..{K_MAX}: inverse-K → {sum(targets.values()) - k2_n:,} layouts "
        f"(TOTAL_N={TOTAL_N:,})"
    )
    if raw_inverse and targets.get(K_INVERSE_MIN) != raw_inverse.get(K_INVERSE_MIN):
        print(
            f"  adjusted K={K_INVERSE_MIN}: {raw_inverse.get(K_INVERSE_MIN)} "
            f"→ {targets.get(K_INVERSE_MIN)} (capacity)"
        )
    print(f"  K={K_INVERSE_MIN}: {targets.get(K_INVERSE_MIN, 0):,}  K=20: {targets.get(K_LOW_MAX, 0):,}  K=50: {targets.get(K_MAX, 0):,}")

    rng = random.Random(SEED)

    if CANDIDATE_POOL_CSV is not None:
        pool_path = Path(CANDIDATE_POOL_CSV).resolve()
        if not pool_path.is_file():
            raise SystemExit(f"Candidate pool missing: {pool_path}")
        pool_rows = load_csv_rows(pool_path)
        pool_by_k: dict[int, list[tuple[int, ...]]] = defaultdict(list)
        for row in pool_rows:
            pool_by_k[row_k(row)].append(row)
        print(f"Candidate pool: {pool_path} ({len(pool_rows):,} rows)")
        selected = sample_from_pool(pool_by_k, targets, exclude=existing_set, rng=rng)
    else:
        print("Candidate pool: random K-hot generation")
        selected = generate_random(targets, exclude=existing_set, rng=rng)

    if len(selected) != sum(targets.values()):
        raise RuntimeError(f"expected {sum(targets.values())} rows, got {len(selected)}")

    overlap = existing_set.intersection(selected)
    if overlap:
        raise RuntimeError(f"{len(overlap)} selected row(s) still overlap existing CSV")

    # Stable sort: K asc, then row lexicographic (matches legacy CSV ordering style)
    selected.sort(key=lambda r: (row_k(r), r))

    write_csv(OUTPUT_CSV, selected)

    by_k = Counter(row_k(r) for r in selected)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "existing_csv": str(EXISTING_CSV),
        "output_csv": str(OUTPUT_CSV),
        "candidate_pool_csv": str(CANDIDATE_POOL_CSV) if CANDIDATE_POOL_CSV else None,
        "total_n": sum(targets.values()),
        "total_n_requested": TOTAL_N,
        "seed": SEED,
        "existing_rows": len(existing_rows),
        "existing_unique": len(existing_set),
        "selected_rows": len(selected),
        "targets_per_k_raw": {
            "2": k2_n,
            **{str(k): raw_inverse[k] for k in sorted(raw_inverse)},
        },
        "include_all_k2": INCLUDE_ALL_K2,
        "k_inverse_range": [K_INVERSE_MIN, K_MAX],
        "targets_per_k": {str(k): targets[k] for k in sorted(targets)},
        "available_per_k": {str(k): available[k] for k in sorted(available)},
        "actual_per_k": {str(k): by_k[k] for k in sorted(by_k)},
        "inverse_k_params": {
            "n_ref": N_REF,
            "tau": TAU,
            "tau_low": TAU_LOW,
            "k_low_max": K_LOW_MAX,
            "k_anchor": K_ANCHOR,
            "n_min": N_MIN,
            "n_max": N_MAX,
            "k_min": K_MIN,
            "k_inverse_min": K_INVERSE_MIN,
            "k_max": K_MAX,
        },
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nWrote {OUTPUT_CSV} ({len(selected):,} rows)")
    print(f"Report  {REPORT_JSON}")
    print(f"K distribution: min={min(by_k.values())} max={max(by_k.values())} buckets={len(by_k)}")


if __name__ == "__main__":
    main()
