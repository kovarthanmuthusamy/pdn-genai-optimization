#!/usr/bin/env python3
"""Sample multi-type decap layout CSVs (type codes, K <= 5).

Primary output (COMBINED mode, default): a **single** CSV mixing three families
of layouts in one file (52 columns, integer type codes per slot):

1. **Type-1 only** — occupied slots are all ``1``.
2. **Type-2 only** — occupied slots are all ``2``.
3. **Mixed types** — each row uses both type ``1`` and type ``2``.

The combined file holds ``TOTAL_COMBINED`` (default 15000) unique random rows,
split evenly across the three families and across K = 2..5, then shuffled.

Schema matches exp060 occupancy (no empty channel in the model tensor;
CSV stores integer codes ``0/1/2``):

    0 = empty, 1 = type-1, 2 = type-2

Legacy per-family CSVs (type-2 only + mixed) are still written when
``WRITE_LEGACY_SPLIT`` is True.

Run:
    python pipelines/heatmaps/sample_multitype_combinations.py
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

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

N_DECAPS = 52
TYPE1 = 1
TYPE2 = 2

# Match type-1 K<=5 layout counts from datasets/data_multifreq_train_K5
# (K hist: {2:1326, 3:1483, 4:1440, 5:1401}, total 5650). No K=1 in that set.
MATCH_TYPE1_K5_COUNTS: dict[int, int] = {2: 1326, 3: 1483, 4: 1440, 5: 1401}

K_MIN_TYPE2 = 2
K_MAX = 5
K_MIN_MIXED = 2

SEED = 42

# If True, both CSVs use MATCH_TYPE1_K5_COUNTS (same amount as type-1 K<=5).
USE_MATCH_TYPE1_COUNTS = True

# Fallback equal allocation when USE_MATCH_TYPE1_COUNTS is False
N_PER_K_TYPE2 = 500
N_PER_K_MIXED = 500
TOTAL_TYPE2: int | None = None
TOTAL_MIXED: int | None = None

OUTPUT_TYPE2_CSV = _ROOT / "data" / "heatmaps" / "combinations_type2_only_K2_5.csv"
OUTPUT_MIXED_CSV = _ROOT / "data" / "heatmaps" / "combinations_mixed_types_K2_5.csv"
REPORT_JSON = _ROOT / "data" / "heatmaps" / "combinations_multitype_sample_report.json"

# --- Combined single-CSV mode (type1-only + type2-only + mixed) ---
# Shape: K=2 fully enumerated (every layout of all 3 families), then the counts
# DECREASE from K=3 to K=5. K=1 (single-type only) is included as its full 52+52
# set so the peak stays at K=2. Total is balanced to exactly TOTAL_COMBINED.
WRITE_COMBINED = True                 # write one merged CSV with all three families
WRITE_LEGACY_SPLIT = False            # also write the two per-family CSVs above
TOTAL_COMBINED = 15000               # total unique rows in the combined CSV
INCLUDE_K1_SINGLE = True             # include all C(52,1) single-type layouts at K=1
K_MAX_COMBINED = K_MAX               # upper K budget (5)
SHUFFLE_COMBINED = True              # shuffle final row order
OUTPUT_COMBINED_CSV = _ROOT / "data" / "heatmaps" / "combinations_multitype_combined_15000.csv"

MAX_GEN_ATTEMPTS_PER_K = 2_000_000

# =============================================================================


def row_k(row: tuple[int, ...] | list[int]) -> int:
    return int(sum(1 for x in row if int(x) != 0))


def is_type1_only(row: tuple[int, ...]) -> bool:
    vals = {int(x) for x in row}
    return vals <= {0, TYPE1} and TYPE1 in vals


def is_type2_only(row: tuple[int, ...]) -> bool:
    vals = {int(x) for x in row}
    return vals <= {0, TYPE2} and TYPE2 in vals


def is_mixed(row: tuple[int, ...]) -> bool:
    vals = {int(x) for x in row}
    return TYPE1 in vals and TYPE2 in vals and vals <= {0, TYPE1, TYPE2}


def write_csv(path: Path, rows: list[tuple[int, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for row in rows:
            w.writerow(list(row))


def targets_per_k(
    *,
    k_min: int,
    k_max: int,
    n_per_k: int,
    total: int | None,
) -> dict[int, int]:
    ks = list(range(k_min, k_max + 1))
    if not ks:
        raise ValueError(f"empty K range {k_min}..{k_max}")
    if total is not None:
        if total < 0:
            raise ValueError(f"total must be >= 0, got {total}")
        base, rem = divmod(int(total), len(ks))
        out = {k: base for k in ks}
        for k in ks[:rem]:
            out[k] += 1
        return out
    if n_per_k < 0:
        raise ValueError(f"n_per_k must be >= 0, got {n_per_k}")
    return {k: int(n_per_k) for k in ks}


def decreasing_per_k(k_start: int, k_max: int, total: int) -> dict[int, int]:
    """Allocate ``total`` over K=k_start..k_max with **decreasing** counts.

    Weights are k_max..1 (largest at k_start), so more layouts land at low K and
    the count tapers down toward k_max. Integer remainder is added to the
    largest (low-K) buckets first, preserving the non-increasing shape.
    """
    ks = list(range(k_start, k_max + 1))
    if not ks:
        return {}
    if total <= 0:
        return {k: 0 for k in ks}
    n = len(ks)
    weights = list(range(n, 0, -1))  # n, n-1, ..., 1
    wsum = sum(weights)
    alloc = [int(total * w // wsum) for w in weights]
    rem = total - sum(alloc)
    i = 0
    while rem > 0:
        alloc[i % n] += 1
        rem -= 1
        i += 1
    return {k: a for k, a in zip(ks, alloc)}


def type2_capacity(k: int) -> int:
    """Distinct type-2-only layouts at budget K (= C(52,K))."""
    return math.comb(N_DECAPS, k)


def mixed_capacity_lower_bound(k: int) -> int:
    """Loose lower bound: choose slots, then 2^K - 2 non-monochrome labelings."""
    if k < 2:
        return 0
    return math.comb(N_DECAPS, k) * (2**k - 2)


def cap_targets(
    targets: dict[int, int],
    capacity_fn,
) -> dict[int, int]:
    """Cap each K by capacity; redistribute shortfall to K with spare room."""
    capped = {k: min(int(targets[k]), int(capacity_fn(k))) for k in targets}
    shortfall = sum(targets.values()) - sum(capped.values())
    if shortfall <= 0:
        return capped
    # Prefer higher K (more capacity) when filling shortfall
    while shortfall > 0:
        candidates = [
            k for k in sorted(capped, reverse=True)
            if capped[k] < capacity_fn(k)
        ]
        if not candidates:
            print(
                f"  warning: could not place remaining {shortfall} rows "
                f"(capacity exhausted); keeping {sum(capped.values())}"
            )
            break
        k = candidates[0]
        capped[k] += 1
        shortfall -= 1
    return capped


def _random_single_type(k: int, rng: random.Random, code: int) -> tuple[int, ...]:
    if k < 1 or k > N_DECAPS:
        raise ValueError(f"invalid K={k}")
    idx = rng.sample(range(N_DECAPS), k)
    row = [0] * N_DECAPS
    for i in idx:
        row[i] = code
    return tuple(row)


def random_type1_only(k: int, rng: random.Random) -> tuple[int, ...]:
    return _random_single_type(k, rng, TYPE1)


def random_type2_only(k: int, rng: random.Random) -> tuple[int, ...]:
    return _random_single_type(k, rng, TYPE2)


# Single-type factories share capacity C(52,K) and exhaustive enumeration.
_SINGLE_TYPE_CODE = {random_type1_only: TYPE1, random_type2_only: TYPE2}


def _all_single_type_k(k: int, code: int) -> list[tuple[int, ...]]:
    """Enumerate every single-type layout (given code) with exactly K slots."""
    from itertools import combinations

    rows: list[tuple[int, ...]] = []
    for idx in combinations(range(N_DECAPS), k):
        row = [0] * N_DECAPS
        for i in idx:
            row[i] = code
        rows.append(tuple(row))
    return rows


def all_type2_only_k(k: int) -> list[tuple[int, ...]]:
    """Enumerate every type-2-only layout with exactly K occupied slots."""
    return _all_single_type_k(k, TYPE2)


def all_mixed_k(k: int) -> list[tuple[int, ...]]:
    """Enumerate every mixed layout (both types present) with K occupied slots.

    For each K-slot choice there are ``2^K - 2`` non-monochrome type labelings.
    Only practical for small K (used for K=2 → C(52,2)*2 = 2652 rows).
    """
    from itertools import combinations, product

    if k < 2:
        return []
    rows: list[tuple[int, ...]] = []
    for idx in combinations(range(N_DECAPS), k):
        for labels in product((TYPE1, TYPE2), repeat=k):
            if len(set(labels)) < 2:  # skip all-type1 / all-type2
                continue
            row = [0] * N_DECAPS
            for i, lab in zip(idx, labels):
                row[i] = lab
            rows.append(tuple(row))
    return rows


def random_mixed(k: int, rng: random.Random) -> tuple[int, ...]:
    """K occupied slots with both types present (rejection if not mixed)."""
    if k < 2:
        raise ValueError("mixed layouts require K >= 2")
    idx = rng.sample(range(N_DECAPS), k)
    # Ensure at least one of each type, then fill remaining uniformly in {1,2}
    n1 = rng.randint(1, k - 1)
    n2 = k - n1
    labels = [TYPE1] * n1 + [TYPE2] * n2
    rng.shuffle(labels)
    row = [0] * N_DECAPS
    for i, lab in zip(idx, labels):
        row[i] = lab
    t = tuple(row)
    if not is_mixed(t):
        raise RuntimeError("internal: mixed generator produced non-mixed row")
    return t


def generate_unique(
    targets: dict[int, int],
    *,
    factory,
    validate,
    rng: random.Random,
    label: str,
) -> list[tuple[int, ...]]:
    selected: list[tuple[int, ...]] = []
    picked: set[tuple[int, ...]] = set()

    for k in sorted(targets):
        need = int(targets[k])
        if need <= 0:
            continue
        # Hard capacity for single-type layouts is C(52,K); mixed is larger but finite.
        single_code = _SINGLE_TYPE_CODE.get(factory)
        if single_code is not None:
            cap = math.comb(N_DECAPS, k)
            if need > cap:
                raise RuntimeError(
                    f"{label}: K={k} need {need} but only C({N_DECAPS},{k})={cap} exist"
                )

        # Exhaustive single-type enumeration when asking for all C(52,K) layouts
        if single_code is not None and need == math.comb(N_DECAPS, k):
            rows_k = _all_single_type_k(k, single_code)
            for row in rows_k:
                if row in picked:
                    continue
                picked.add(row)
                selected.append(row)
            print(f"  {label} K={k}: enumerated all {len(rows_k):,} layouts")
            continue

        got = 0
        attempts = 0
        while got < need:
            attempts += 1
            if attempts > MAX_GEN_ATTEMPTS_PER_K:
                raise RuntimeError(
                    f"{label}: K={k} could not reach {need} unique rows "
                    f"(got {got} after {MAX_GEN_ATTEMPTS_PER_K} attempts)"
                )
            row = factory(k, rng)
            if not validate(row):
                continue
            if row in picked:
                continue
            picked.add(row)
            selected.append(row)
            got += 1

    selected.sort(key=lambda r: (row_k(r), r))
    return selected


def _summarize(rows: list[tuple[int, ...]]) -> dict:
    by_k = Counter(row_k(r) for r in rows)
    type_hist = Counter()
    for r in rows:
        for v in r:
            if v != 0:
                type_hist[int(v)] += 1
    return {
        "rows": len(rows),
        "unique": len(set(rows)),
        "per_k": {str(k): by_k[k] for k in sorted(by_k)},
        "nonzero_type_counts": {str(t): type_hist[t] for t in sorted(type_hist)},
    }


def build_combined(rng: random.Random) -> tuple[list[tuple[int, ...]], dict]:
    """Generate one merged list of TOTAL_COMBINED rows with the target K-shape.

    Shape:
      * K=1 : all C(52,1) single-type layouts (52 type-1 + 52 type-2), if enabled.
      * K=2 : **every** layout of all three families (fully enumerated) — the peak.
      * K=3..K_MAX : totals **decrease** with K; each K split across the three
        families; sampled uniquely at random.

    The K=3..K_MAX budget is whatever remains after the fixed K=1/K=2 sets, so the
    grand total lands exactly on TOTAL_COMBINED.
    """
    families = ("type1_only", "type2_only", "mixed")
    sample_specs = {
        "type1_only": (random_type1_only, is_type1_only),
        "type2_only": (random_type2_only, is_type2_only),
        "mixed": (random_mixed, is_mixed),
    }

    fam_rows: dict[str, list[tuple[int, ...]]] = {fam: [] for fam in families}

    # --- K=1: full single-type sets (no mixed at K=1) ---
    if INCLUDE_K1_SINGLE:
        fam_rows["type1_only"].extend(_all_single_type_k(1, TYPE1))
        fam_rows["type2_only"].extend(_all_single_type_k(1, TYPE2))
        print(f"K=1: enumerated all single-type layouts "
              f"(type1={len(_all_single_type_k(1, TYPE1))}, "
              f"type2={len(_all_single_type_k(1, TYPE2))})")

    # --- K=2: enumerate EVERY layout of all three families ---
    k2_t1 = _all_single_type_k(2, TYPE1)
    k2_t2 = _all_single_type_k(2, TYPE2)
    k2_mix = all_mixed_k(2)
    fam_rows["type1_only"].extend(k2_t1)
    fam_rows["type2_only"].extend(k2_t2)
    fam_rows["mixed"].extend(k2_mix)
    k2_total = len(k2_t1) + len(k2_t2) + len(k2_mix)
    print(f"K=2: enumerated ALL layouts "
          f"(type1={len(k2_t1)}, type2={len(k2_t2)}, mixed={len(k2_mix)}, total={k2_total})")

    fixed = sum(len(v) for v in fam_rows.values())
    remaining = int(TOTAL_COMBINED) - fixed
    if remaining < 0:
        raise ValueError(
            f"TOTAL_COMBINED={TOTAL_COMBINED} < fixed K<=2 rows={fixed}; raise TOTAL_COMBINED"
        )

    # --- K=3..K_MAX: decreasing totals, split across families ---
    kdist = decreasing_per_k(3, K_MAX_COMBINED, remaining)
    fam_targets: dict[str, dict[int, int]] = {fam: {} for fam in families}
    for k in sorted(kdist):
        base, r = divmod(int(kdist[k]), len(families))
        for i, fam in enumerate(families):
            fam_targets[fam][k] = base + (1 if i < r else 0)
    print(f"K=3..{K_MAX_COMBINED}: remaining={remaining}, decreasing totals={kdist}")

    for fam in families:
        targets = {k: v for k, v in fam_targets[fam].items() if v > 0}
        if not targets:
            continue
        factory, validate = sample_specs[fam]
        rows = generate_unique(
            targets,
            factory=factory,
            validate=validate,
            rng=rng,
            label=fam,
        )
        fam_rows[fam].extend(rows)

    combined: list[tuple[int, ...]] = []
    report_families: dict = {}
    for fam in families:
        combined.extend(fam_rows[fam])
        report_families[fam] = {
            "targets_k3plus": {str(k): fam_targets[fam][k] for k in sorted(fam_targets[fam])},
            **_summarize(fam_rows[fam]),
        }

    # Families cannot collide (distinct nonzero value sets) and K differs; assert.
    if len(set(combined)) != len(combined):
        raise RuntimeError("unexpected duplicate rows in combined set")
    if len(combined) != int(TOTAL_COMBINED):
        raise RuntimeError(
            f"combined rows {len(combined)} != TOTAL_COMBINED {TOTAL_COMBINED}"
        )

    if SHUFFLE_COMBINED:
        rng.shuffle(combined)

    return combined, report_families


def main() -> None:
    rng = random.Random(SEED)

    if WRITE_COMBINED:
        combined, report_families = build_combined(rng)
        write_csv(OUTPUT_COMBINED_CSV, combined)
        print(f"\nwrote {OUTPUT_COMBINED_CSV} ({len(combined):,} rows)")

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "n_decaps": N_DECAPS,
            "mode": "combined",
            "schema": {"empty": 0, "type1": TYPE1, "type2": TYPE2},
            "combined": {
                "output_csv": str(OUTPUT_COMBINED_CSV),
                "total_target": TOTAL_COMBINED,
                "k_range": [1 if INCLUDE_K1_SINGLE else 2, K_MAX_COMBINED],
                "k2_fully_enumerated": True,
                "include_k1_single": INCLUDE_K1_SINGLE,
                "shuffled": SHUFFLE_COMBINED,
                "families": report_families,
                **_summarize(combined),
            },
        }
        REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
        REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report {REPORT_JSON}")
        if not WRITE_LEGACY_SPLIT:
            print("Done.")
            return

    if USE_MATCH_TYPE1_COUNTS:
        t2_raw = {k: MATCH_TYPE1_K5_COUNTS[k] for k in range(K_MIN_TYPE2, K_MAX + 1)}
        mixed_raw = {k: MATCH_TYPE1_K5_COUNTS[k] for k in range(K_MIN_MIXED, K_MAX + 1)}
        print(f"Using type-1 K5 matched counts: {MATCH_TYPE1_K5_COUNTS} (total={sum(MATCH_TYPE1_K5_COUNTS.values())})")
    else:
        t2_raw = targets_per_k(
            k_min=K_MIN_TYPE2,
            k_max=K_MAX,
            n_per_k=N_PER_K_TYPE2,
            total=TOTAL_TYPE2,
        )
        mixed_raw = targets_per_k(
            k_min=K_MIN_MIXED,
            k_max=K_MAX,
            n_per_k=N_PER_K_MIXED,
            total=TOTAL_MIXED,
        )
    t2_targets = cap_targets(t2_raw, type2_capacity)
    mixed_targets = cap_targets(mixed_raw, mixed_capacity_lower_bound)

    print("=== Type-2 only ===")
    if t2_targets != t2_raw:
        print(f"  raw targets: {t2_raw}")
        print(f"  capped by C(52,K): {t2_targets}  (sum={sum(t2_targets.values())})")
    else:
        print(f"  targets: {t2_targets}  (sum={sum(t2_targets.values())})")
    type2_rows = generate_unique(
        t2_targets,
        factory=random_type2_only,
        validate=is_type2_only,
        rng=rng,
        label="type2_only",
    )
    write_csv(OUTPUT_TYPE2_CSV, type2_rows)
    print(f"  wrote {OUTPUT_TYPE2_CSV} ({len(type2_rows):,} rows)")

    print("=== Mixed types (1 & 2) ===")
    if mixed_targets != mixed_raw:
        print(f"  raw targets: {mixed_raw}")
        print(f"  capped: {mixed_targets}  (sum={sum(mixed_targets.values())})")
    else:
        print(f"  targets: {mixed_targets}  (sum={sum(mixed_targets.values())})")
    mixed_rows = generate_unique(
        mixed_targets,
        factory=random_mixed,
        validate=is_mixed,
        rng=rng,
        label="mixed",
    )
    write_csv(OUTPUT_MIXED_CSV, mixed_rows)
    print(f"  wrote {OUTPUT_MIXED_CSV} ({len(mixed_rows):,} rows)")

    # Cross-file uniqueness (type-2-only vs mixed cannot collide by construction,
    # but check anyway).
    overlap = set(type2_rows).intersection(mixed_rows)
    if overlap:
        raise RuntimeError(f"unexpected overlap between type2 and mixed: {len(overlap)}")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "n_decaps": N_DECAPS,
        "schema": {
            "empty": 0,
            "type1": TYPE1,
            "type2": TYPE2,
            "note": "CSV stores integer type codes; empty has no dedicated channel",
        },
        "type2_only": {
            "output_csv": str(OUTPUT_TYPE2_CSV),
            "k_range": [K_MIN_TYPE2, K_MAX],
            "targets_per_k_raw": {str(k): t2_raw[k] for k in sorted(t2_raw)},
            "targets_per_k": {str(k): t2_targets[k] for k in sorted(t2_targets)},
            **_summarize(type2_rows),
        },
        "mixed": {
            "output_csv": str(OUTPUT_MIXED_CSV),
            "k_range": [K_MIN_MIXED, K_MAX],
            "targets_per_k_raw": {str(k): mixed_raw[k] for k in sorted(mixed_raw)},
            "targets_per_k": {str(k): mixed_targets[k] for k in sorted(mixed_targets)},
            **_summarize(mixed_rows),
        },
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport {REPORT_JSON}")
    print("Done.")


if __name__ == "__main__":
    main()
