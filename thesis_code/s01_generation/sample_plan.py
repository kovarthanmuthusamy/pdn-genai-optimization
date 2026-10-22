"""Plan the 30,000 placements of the dataset and the frequencies of their maps.

What: the dataset grows a first balanced set (every placement kept) to 30,000 placements: K = 1 keeps
    all 156 placements that exist, K = 2..10 grow to 3,316 each. At every K the one-, two- and
    three-part classes have equal shares (K = 2: one- and two-part), and inside a class every part or
    pair of parts has an equal share. For K >= 6, slot "C1" is mounted on K/52 of each class; those
    placements are always simulated new. Every placement has exactly 11 maps: one at 200 MHz, one
    below 10 MHz, one above 500 MHz and eight in 10-500 MHz (FORM).
    Extra placements come first from the set-aside pool of the first set (already simulated, never
    trained on), with the maps each still needs ("fill" campaign). The rest is drawn new and
    simulated with spectrum and 11 maps ("new" campaign). A new placement is absent from the first
    set, the pool, every older store and every independent test placement.
Usage: python -m thesis_code.s01_generation.sample_plan
    Writes to OUTPUT_DIR/s01_generation/:
        balance30k_pool_ids.csv               pool placements moved into the dataset
        balance30k_setaside_ids.csv           pool placements left aside (the test pool)
        balance30k_drop_samples_pool.csv      their rows that repeat a frequency of the same placement
        combinations_balance30k_new*          new placements: type codes, 11 map frequencies, ids
        combinations_balance30k_fill*         pool placements: the maps each still needs (NaN-padded)
        balance30k_planned_maps.csv           the whole dataset, one row per map -- checked against FORM
        balance30k_report.json                per-K / per-class table, map counts, batches
Config keys: V1_RAW, POOL, OTHER_STORES, UNSEEN_MANIFESTS_GLOB, PER_K, STEM, C1_NEW_FROM_K, SEED_*, BATCH.
"""
from __future__ import annotations

import csv
import glob
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from thesis_code.common import settings
from thesis_code.common.paths import out_dir

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
V1_RAW = settings.REPO / "datasets/pdn_k1-10_balanced_v1/raw"                  # first balanced set, all kept
POOL = settings.REPO / "datasets/pdn_k1-10_balanced_v1/test_pool/raw"          # its set-aside pool, simulated
OTHER_STORES = [settings.REPO / "datasets/data_multifreq_k1_k10_randfreq",      # novelty check only
                settings.REPO / "datasets/data_multifreq_k6_k10"]
UNSEEN_MANIFESTS_GLOB = "experiments/*/unseen*/unseen_manifest.json"            # independent placements stay unseen
PER_K = {1: 156, **{k: 3316 for k in range(2, 11)}}
STEM = "balance30k"
C1_NEW_FROM_K = 6
SEED_SELECT, SEED_NEW, SEED_FILL = 61, 62, 65
BATCH = 2000                                          # groups per eCADSTAR batch
# =============================================================================

N_DECAPS = 52
PI_REF_MHZ = 200.0
LOW_BAND, HIGH_BAND = (1.0, 10.0), (500.0, 600.0)
FORM = (1, 1, 1, 8)               # maps at 200 MHz, below 10 MHz, above 500 MHz, in 10-500 MHz
N_MID = FORM[3]
MID_LOW, MID_HIGH = (10.0, 100.0), (100.0, 500.0)
SEC_PER_GROUP = 0.44              # measured eCADSTAR time per simulated group
SUBS = {1: [(1,), (2,), (3,)], 2: [(1, 2), (1, 3), (2, 3)], 3: [(1, 2, 3)]}
PART = {1: "100nF", 2: "47nF", 3: "22nF"}
# One map of each new placement is drawn from this frequency grid.
ANCHORS_MHZ = [10.0, 63.0, 80.0, 100.0, 120.0, 150.0, 170.0, 180.0, 200.0, 230.0,
               250.0, 270.0, 280.0, 300.0, 330.0, 350.0, 370.0, 390.0, 400.0, 420.0,
               430.0, 450.0, 470.0, 500.0]


def store_layouts(store: Path) -> tuple[list[str], np.ndarray]:
    with (store / "manifest.csv").open() as fh:
        ids = sorted({r["design_id"] for r in csv.DictReader(fh)})
    codes = np.stack([np.load(store / "layouts" / d / "type.npy").astype(int).reshape(-1) for d in ids])
    return ids, codes


def parts_of(row: np.ndarray) -> tuple[int, ...]:
    return tuple(t for t in (1, 2, 3) if (row == t).any())


def split_even(n: int, parts: int) -> list[int]:
    base, rem = divmod(n, parts)
    return [base + (i < rem) for i in range(parts)]


def low_draw(rng: np.random.Generator, n: int) -> np.ndarray:
    return np.clip(np.round(rng.uniform(*LOW_BAND, n), 1), 1.0, 9.9)


def mid_draw(rng: np.random.Generator, shape, band: tuple[float, float]) -> np.ndarray:
    lo, hi = band
    return np.clip(np.round(rng.uniform(lo, hi, shape), 1), lo, hi - 0.1)


def form_of(freqs: list[float]) -> tuple[int, int, int, int]:
    f = np.asarray(freqs, dtype=float)
    return (int((f == PI_REF_MHZ).sum()), int((f < 10.0).sum()), int((f > 500.0).sum()),
            int(((f >= 10.0) & (f <= 500.0) & (f != PI_REF_MHZ)).sum()))


def fill_plan(rng: np.random.Generator, cur: list[float]) -> list[float]:
    """Extra frequencies that bring a placement with maps at ``cur`` to FORM (none may be in excess)."""
    ref, low, high, mid = form_of(cur)
    if ref > 1 or low > 1 or high > 1 or mid > N_MID:
        raise SystemExit(f"layout with maps {sorted(cur)} has more than the form allows")
    plan: list[float] = [PI_REF_MHZ] * (1 - ref)
    plan += [float(low_draw(rng, 1)[0])] * (1 - low)
    plan += [float(np.clip(np.round(rng.uniform(*HIGH_BAND), 1), 500.1, 600.0))] * (1 - high)
    need = N_MID - mid
    if need and not any(MID_LOW[0] <= f < MID_LOW[1] for f in cur):
        plan += [float(mid_draw(rng, 1, MID_LOW)[0])]
        need -= 1
    plan += mid_draw(rng, need, MID_HIGH).tolist()
    while len(set(plan) | set(cur)) < len(plan) + len(set(cur)):   # redraw collisions in 100-500
        plan = plan[:len(plan) - need] + mid_draw(rng, need, MID_HIGH).tolist()
    return plan


def rows_by_design(store: Path) -> dict[str, list[tuple[str, float]]]:
    out: dict[str, list[tuple[str, float]]] = {}
    with (store / "manifest.csv").open() as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["design_id"], []).append((r["sample_name"], float(r["freq_mhz"])))
    return out


def write_or_verify(out: Path, stem: str, codes: np.ndarray, freqs: np.ndarray, ids: list[str]) -> str:
    """Write <stem>.csv / _freqs.csv / _design_ids.csv; if they exist they must be identical."""
    paths = [out / f"{stem}.csv", out / f"{stem}_freqs.csv", out / f"{stem}_design_ids.csv"]
    tmp = [p.with_suffix(p.suffix + ".tmp") for p in paths]
    np.savetxt(tmp[0], codes, delimiter=",", fmt="%d")
    np.savetxt(tmp[1], freqs, delimiter=",", fmt="%.1f")
    tmp[2].write_text("\n".join(ids) + "\n")
    if all(p.exists() for p in paths):
        same = all(p.read_bytes() == t.read_bytes() for p, t in zip(paths, tmp))
        for t in tmp:
            t.unlink()
        if not same:
            raise SystemExit(f"{stem}: files on disk differ from this sampler's output -- not overwritten")
        return "verified identical to files on disk"
    for p, t in zip(paths, tmp):
        t.replace(p)
    return "written"


def new_layout(rng: np.random.Generator, k: int, sub: tuple[int, ...], c1: str) -> np.ndarray:
    """One placement with exactly the parts in ``sub``. c1: 'yes' / 'no' / 'any' (slot "C1" = index 0)."""
    row = np.zeros(N_DECAPS, dtype=int)
    if c1 == "yes":
        slots = np.concatenate([[0], 1 + rng.choice(N_DECAPS - 1, size=k - 1, replace=False)])
    elif c1 == "no":
        slots = 1 + rng.choice(N_DECAPS - 1, size=k, replace=False)
    else:
        slots = rng.choice(N_DECAPS, size=k, replace=False)
    while True:                                   # each decap's part at random among ``sub``, all present
        types = rng.choice(sub, size=k)
        if len(set(types.tolist())) == len(sub):
            break
    row[slots] = types
    return row


def main() -> None:
    out = out_dir("s01_generation")
    assert sum(PER_K.values()) == 30_000, sum(PER_K.values())
    vids, vcodes = store_layouts(V1_RAW)
    pids, pcodes = store_layouts(POOL)
    vK, pK = (vcodes > 0).sum(1), (pcodes > 0).sum(1)
    vsub, psub = [parts_of(r) for r in vcodes], [parts_of(r) for r in pcodes]
    assert not set(vids) & set(pids), "pool overlaps v1"

    seen = {c.tobytes() for c in vcodes} | {c.tobytes() for c in pcodes}
    for st in OTHER_STORES:
        seen |= {c.tobytes() for c in store_layouts(st)[1]}
    n_unseen = 0
    for mf in glob.glob(str(settings.REPO / UNSEEN_MANIFESTS_GLOB)):
        rows = json.loads(Path(mf).read_text())["rows"]
        n_unseen += len(rows)
        seen |= {np.asarray(r["type_codes"], dtype=int).tobytes() for r in rows}

    rng_sel, rng_new = np.random.default_rng(SEED_SELECT), np.random.default_rng(SEED_NEW)
    pool_idx: list[int] = []
    new_rows: list[np.ndarray] = []
    table: list[dict] = []
    for k in sorted(PER_K):
        classes = [1] if k == 1 else [1, 2] if k == 2 else [1, 2, 3]
        for n_class, cls in zip(split_even(PER_K[k], len(classes)), classes):
            for target, sub in zip(split_even(n_class, len(SUBS[cls])), SUBS[cls]):
                in_v1 = [i for i in range(len(vids)) if vK[i] == k and vsub[i] == sub]
                v1_c1 = sum(int(vcodes[i][0] > 0) for i in in_v1)
                add = target - len(in_v1)
                if add < 0:
                    raise SystemExit(f"K={k} {sub}: v1 already has {len(in_v1)} > target {target}")
                n_c1 = max(0, round(target * k / N_DECAPS) - v1_c1) if k >= C1_NEW_FROM_K else 0
                n_plain = add - n_c1
                elig = [i for i in range(len(pids)) if pK[i] == k and psub[i] == sub
                        and (k < C1_NEW_FROM_K or pcodes[i][0] == 0)]
                take = [int(i) for i in rng_sel.permutation(np.array(elig, dtype=int))[:n_plain]]
                n_new_plain = n_plain - len(take)
                for c1, n in (("yes", n_c1), ("no" if k >= C1_NEW_FROM_K else "any", n_new_plain)):
                    made = 0
                    while made < n:
                        row = new_layout(rng_new, k, sub, c1)
                        if row.tobytes() in seen:
                            continue
                        seen.add(row.tobytes()); new_rows.append(row); made += 1
                pool_idx += take
                table.append({"K": k, "class": f"{cls}-part", "parts": "+".join(PART[t] for t in sub),
                              "target": target, "v1": len(in_v1), "add": add, "pool_eligible": len(elig),
                              "from_pool": len(take), "new": n_c1 + n_new_plain, "new_with_c1": n_c1})

    pool_idx.sort()
    pool_ids = [pids[i] for i in pool_idx]
    (out / f"{STEM}_pool_ids.csv").write_text("\n".join(pool_ids) + "\n")
    taken = set(pool_ids)                 # the rest of the pool stays aside as the test pool
    (out / f"{STEM}_setaside_ids.csv").write_text("\n".join(d for d in pids if d not in taken) + "\n")

    # ---- new placements: shuffled so every batch mixes K, classes and C1; the 11-map form ------------
    new = np.stack(new_rows)[rng_new.permutation(len(new_rows))]
    n = len(new)
    anchors = [a for a in ANCHORS_MHZ if a != PI_REF_MHZ]
    fr = np.concatenate([rng_new.choice(anchors, size=(n, 1)),
                         mid_draw(rng_new, (n, 1), MID_LOW),
                         mid_draw(rng_new, (n, N_MID - 2), MID_HIGH),
                         np.full((n, 1), PI_REF_MHZ), low_draw(rng_new, n)[:, None],
                         np.clip(np.round(rng_new.uniform(*HIGH_BAND, (n, 1)), 1), 500.1, 600.0)], 1)
    for i in range(n):
        while len(set(fr[i])) < fr.shape[1]:
            fr[i, 1:2] = mid_draw(rng_new, 1, MID_LOW)
            fr[i, 2:N_MID] = mid_draw(rng_new, N_MID - 2, MID_HIGH)
    new_ids = [f"b30_d{i:06d}" for i in range(n)]
    st_new = write_or_verify(out, f"combinations_{STEM}_new", new, fr, new_ids)

    # ---- pool placements: drop repeated frequencies, fill up to the form ------------------------------
    prow = rows_by_design(POOL)
    rng = np.random.default_rng(SEED_FILL)
    drop: list[str] = []
    fill_ids, fill_codes, fill_fr = [], [], []
    kept_pool_rows: dict[str, list[float]] = {}
    for i in pool_idx:
        d, cur = pids[i], []
        for s, f in prow[d]:
            if f in cur:
                drop.append(s)
                continue
            cur.append(f)
        kept_pool_rows[d] = cur
        plan = fill_plan(rng, cur)
        if plan:
            fill_ids.append(d); fill_codes.append(pcodes[i]); fill_fr.append(plan)
    if fill_fr:
        width = max(len(x) for x in fill_fr)
        fill_mat = np.full((len(fill_fr), width), np.nan)
        for j, x in enumerate(fill_fr):
            fill_mat[j, :len(x)] = x
        st_fill = write_or_verify(out, f"combinations_{STEM}_fill", np.stack(fill_codes), fill_mat, fill_ids)
    else:
        fill_mat, st_fill = np.zeros((0, 0)), "nothing to fill"
    (out / f"{STEM}_drop_samples_pool.csv").write_text("\n".join(drop) + ("\n" if drop else ""))

    # ---- the whole plan, one row per map; every placement must have the form --------------------------
    planned: list[tuple[str, float, str]] = []
    for d, rows in rows_by_design(V1_RAW).items():
        planned += [(d, f, "v1") for _, f in rows]
    for d, cur in kept_pool_rows.items():
        planned += [(d, f, "set-aside pool") for f in cur]
    planned += [(d, float(f), "fill campaign") for d, x in zip(fill_ids, fill_fr) for f in x]
    planned += [(d, float(f), "new campaign") for d, row in zip(new_ids, fr) for f in row]
    per_layout: dict[str, list[float]] = {}
    for d, f, _ in planned:
        per_layout.setdefault(d, []).append(f)
    bad = {d: v for d, v in per_layout.items() if form_of(v) != FORM or len(set(v)) != len(v)}
    if bad or len(per_layout) != sum(PER_K.values()):
        raise SystemExit(f"{len(bad)} layouts break the form, e.g. {list(bad.items())[:2]}; "
                         f"{len(per_layout)} layouts planned")
    with (out / f"{STEM}_planned_maps.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["design_id", "freq_mhz", "status"])
        w.writerows((d, f"{f:.1f}", s) for d, f, s in planned)

    per_k = {}
    for k in sorted(PER_K):
        rows = [t for t in table if t["K"] == k]
        per_k[k] = {key: sum(t[key] for t in rows) for key in ("target", "v1", "add", "from_pool", "new", "new_with_c1")}
    n_fill = int(np.isfinite(fill_mat).sum())
    batches = {"new_impedance": math.ceil(n / BATCH), "new_heatmap": math.ceil(fr.size / BATCH),
               "fill_heatmap": math.ceil(n_fill / BATCH)}
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "v1": str(V1_RAW.relative_to(settings.REPO)), "pool": str(POOL.relative_to(settings.REPO)),
              "per_k_target": PER_K, "per_k": per_k, "cells": table,
              "totals": {"layouts": len(per_layout), "v1": len(vids), "from_pool": len(pool_ids), "new": n,
                         "pool_left_for_testing": len(pids) - len(pool_ids)},
              "unseen_boards_excluded": n_unseen,
              "heatmaps": {"new": int(fr.size), "fill": n_fill, "dropped_pool_duplicates": len(drop)},
              "batches": batches, "expected_merged_rows": len(planned),
              "status": {"new": st_new, "fill": st_fill},
              "seeds": {"select": SEED_SELECT, "new": SEED_NEW, "fill": SEED_FILL},
              "estimated_hours": round((fr.size + n_fill + n) * SEC_PER_GROUP / 3600, 1)}
    (out / f"{STEM}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{'K':>2s} {'target':>7s} {'v1':>6s} {'add':>5s} {'pool':>5s} {'new':>5s} {'new w/ C1':>9s}")
    for k, v in per_k.items():
        print(f"{k:2d} {v['target']:7,d} {v['v1']:6,d} {v['add']:5,d} {v['from_pool']:5,d} {v['new']:5,d} {v['new_with_c1']:9,d}")
    t = report["totals"]
    print(f"   {t['layouts']:7,d} {t['v1']:6,d} {t['from_pool'] + t['new']:5,d} {t['from_pool']:5,d} {t['new']:5,d}")
    print(f"new  : {n:,} placements x {fr.shape[1]} maps = {fr.size:,} maps + {n:,} spectra  [{st_new}]")
    print(f"fill : {len(fill_ids):,} pool placements, {n_fill:,} maps  [{st_fill}]")
    print(f"pool : {len(pool_ids):,} set-aside placements moved in ({len(drop)} repeated-frequency rows dropped); "
          f"{t['pool_left_for_testing']:,} stay in the test pool")
    print(f"batches of {BATCH}: {batches}; merged store {len(planned):,} rows; about {report['estimated_hours']} h")


if __name__ == "__main__":
    main()
