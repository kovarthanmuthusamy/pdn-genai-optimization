"""Random decap-layout × PI-frequency candidate pool generation.

Run:
    python active_learning_pi/al/candidates.py"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class Candidate:
    candidate_id: int
    occupancy: list[int]  # length 52
    mhz: float
    k: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def random_occupancy(k: int, rng: np.random.Generator, n_decaps: int = 52) -> np.ndarray:
    occ = np.zeros(n_decaps, dtype=np.int8)
    if k <= 0:
        return occ
    idx = rng.choice(n_decaps, size=min(k, n_decaps), replace=False)
    occ[idx] = 1
    return occ


def build_mhz_schedule(
    strata: dict[float, int],
    num_candidates: int,
    rng: np.random.Generator,
) -> list[float]:
    """Expand MHz→count strata to a shuffled per-candidate MHz list (length ``num_candidates``)."""
    schedule: list[float] = []
    for mhz, count in sorted(strata.items(), key=lambda x: x[0]):
        schedule.extend([float(mhz)] * int(count))
    if len(schedule) != num_candidates:
        raise ValueError(
            f"mhz_strata counts sum to {len(schedule)}, expected {num_candidates}: {strata}"
        )
    rng.shuffle(schedule)
    return schedule


def _scale_strata_to_target(strata: dict[float, int], target_n: int) -> dict[float, int]:
    """Scale MHz→count strata to sum to ``target_n`` while preserving proportions."""
    if target_n < 0:
        raise ValueError(f"target_n must be >= 0, got {target_n}")
    if target_n == 0:
        return {}
    total = int(sum(int(v) for v in strata.values()))
    if total <= 0:
        raise ValueError("mhz_strata total must be > 0 to scale")
    # Largest remainder method
    items = [(float(k), float(int(v)) * target_n / total) for k, v in strata.items()]
    base = {k: int(np.floor(x)) for k, x in items}
    remainder = target_n - sum(base.values())
    if remainder > 0:
        frac = sorted(((k, x - base[k]) for k, x in items), key=lambda t: t[1], reverse=True)
        for i in range(remainder):
            base[frac[i % len(frac)][0]] += 1
    # Clean zeros
    return {k: v for k, v in base.items() if v > 0}


def _quantize_mhz(x: float, step: float) -> float:
    if step <= 0:
        return float(x)
    return float(round(float(x) / float(step)) * float(step))


def _sample_explore_mhz(
    *,
    rng: np.random.Generator,
    n: int,
    explore_mhz_grid: list[float] | None = None,
    explore_mhz_min: float | None = None,
    explore_mhz_max: float | None = None,
    explore_mhz_quantize: float = 0.0,
    explore_mhz_band_edges: list[float] | None = None,
    avoid_mhz: set[float] | None = None,
) -> list[float]:
    if n <= 0:
        return []
    if explore_mhz_grid:
        arr = np.array([float(x) for x in explore_mhz_grid], dtype=np.float64)
        return [float(rng.choice(arr)) for _ in range(n)]

    if explore_mhz_min is None or explore_mhz_max is None:
        raise ValueError("explore_mhz_min/max must be set when explore_mhz_grid is not provided")
    lo = float(explore_mhz_min)
    hi = float(explore_mhz_max)
    if hi <= lo:
        raise ValueError(f"explore_mhz_max must be > explore_mhz_min, got {lo}..{hi}")

    avoid = {round(float(x), 6) for x in (avoid_mhz or set())}
    out: list[float] = []
    edges = [float(x) for x in (explore_mhz_band_edges or [])]
    if len(edges) >= 2:
        edges = sorted(edges)
        # Clip edges to [lo, hi]
        edges = [max(lo, min(hi, e)) for e in edges]
        # Build valid bands
        bands: list[tuple[float, float]] = []
        for a, b in zip(edges[:-1], edges[1:]):
            if b > a:
                bands.append((a, b))
        if not bands:
            raise ValueError("explore_mhz_band_edges produced no valid bands")
        for _ in range(n):
            a, b = bands[int(rng.integers(0, len(bands)))]
            # Resample a few times to avoid landing on main MHz after quantization
            val = None
            for _tries in range(20):
                x = float(rng.uniform(a, b))
                q = _quantize_mhz(x, explore_mhz_quantize)
                if round(float(q), 6) not in avoid:
                    val = q
                    break
            out.append(float(val if val is not None else _quantize_mhz(float(rng.uniform(a, b)), explore_mhz_quantize)))
    else:
        for _ in range(n):
            val = None
            for _tries in range(20):
                x = float(rng.uniform(lo, hi))
                q = _quantize_mhz(x, explore_mhz_quantize)
                if round(float(q), 6) not in avoid:
                    val = q
                    break
            out.append(float(val if val is not None else _quantize_mhz(float(rng.uniform(lo, hi)), explore_mhz_quantize)))

    # Ensure within bounds after quantization
    out = [max(lo, min(hi, float(x))) for x in out]
    return out


def _mhz_key(mhz: float) -> float:
    return round(float(mhz), 6)


def generate_candidates(
    *,
    num_candidates: int,
    mhz_grid: list[float],
    fixed_k: int | None = None,
    k_values: list[int] | None = None,
    seed: int = 42,
    mhz_priority: list[float] | None = None,
    mhz_strata: dict[float, int] | None = None,
    explore_mhz_grid: list[float] | None = None,
    explore_n: int = 0,
    explore_mhz_min: float | None = None,
    explore_mhz_max: float | None = None,
    explore_mhz_quantize: float = 0.0,
    explore_mhz_band_edges: list[float] | None = None,
) -> list[Candidate]:
    """
    Sample (occupancy, MHz) pairs: random K-hot decap layouts × frequencies.

    If ``mhz_strata`` is set (MHz → count), candidates get MHz from that schedule; occupancy
    still varies randomly.

    If ``explore_n > 0``, we reserve ``explore_n`` slots for exploration MHz sampled uniformly
    either from ``explore_mhz_grid`` or from ``[explore_mhz_min, explore_mhz_max]`` (optionally
    quantized and/or band-stratified via ``explore_mhz_band_edges``). The exploration slice is
    shuffled into the pool.
    The remaining ``num_candidates - explore_n`` are drawn from the (optionally scaled) strata.
    Otherwise MHz is drawn randomly from ``mhz_grid`` (legacy).
    """
    rng = np.random.default_rng(seed)
    ks = sorted({int(k) for k in (k_values or [fixed_k or 5])})
    k_schedule = [ks[i % len(ks)] for i in range(num_candidates)]
    rng.shuffle(k_schedule)

    explore_n = int(explore_n or 0)
    if explore_n < 0 or explore_n > num_candidates:
        raise ValueError(f"explore_n must be in [0, {num_candidates}], got {explore_n}")

    mhz_schedule: list[float]
    if mhz_strata:
        base_n = num_candidates - explore_n
        if sum(int(v) for v in mhz_strata.values()) == base_n:
            base_strata = mhz_strata
        else:
            base_strata = _scale_strata_to_target(mhz_strata, base_n)
        mhz_schedule = build_mhz_schedule(base_strata, base_n, rng)
        if explore_n > 0:
            mhz_schedule.extend(
                _sample_explore_mhz(
                    rng=rng,
                    n=explore_n,
                    explore_mhz_grid=explore_mhz_grid,
                    explore_mhz_min=explore_mhz_min,
                    explore_mhz_max=explore_mhz_max,
                    explore_mhz_quantize=float(explore_mhz_quantize or 0.0),
                    explore_mhz_band_edges=explore_mhz_band_edges,
                    avoid_mhz=set(base_strata.keys()),
                )
            )
            rng.shuffle(mhz_schedule)
    else:
        # Legacy random draw (optionally biased by mhz_priority)
        mhz_pool = list(mhz_grid)
        if mhz_priority:
            for m in mhz_priority:
                if m in mhz_grid or m not in mhz_pool:
                    mhz_pool.extend([m] * 2)
        if explore_n > 0:
            try:
                extra = _sample_explore_mhz(
                    rng=rng,
                    n=max(1, explore_n // 2),
                    explore_mhz_grid=explore_mhz_grid,
                    explore_mhz_min=explore_mhz_min,
                    explore_mhz_max=explore_mhz_max,
                    explore_mhz_quantize=float(explore_mhz_quantize or 0.0),
                    explore_mhz_band_edges=explore_mhz_band_edges,
                )
                mhz_pool.extend(extra)
            except Exception:
                # Keep legacy behavior if exploration config is incomplete in legacy mode
                pass
        mhz_arr = np.array(mhz_pool, dtype=np.float64)
        mhz_schedule = [float(rng.choice(mhz_arr)) for _ in range(num_candidates)]

    out: list[Candidate] = []
    for cid, k in enumerate(k_schedule):
        mhz = mhz_schedule[cid]
        occ = random_occupancy(k, rng)
        out.append(
            Candidate(
                candidate_id=cid,
                occupancy=occ.astype(int).tolist(),
                mhz=mhz,
                k=k,
            )
        )
    return out
