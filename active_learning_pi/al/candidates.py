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


def generate_candidates(
    *,
    num_candidates: int,
    mhz_grid: list[float],
    fixed_k: int,
    seed: int = 42,
    mhz_priority: list[float] | None = None,
) -> list[Candidate]:
    """
    Sample (occupancy, MHz) pairs: random K-hot decap layouts × frequencies from grid.
    Oversamples priority MHz (e.g. 80, 250) by duplicating those frequencies in the pool.
    """
    rng = np.random.default_rng(seed)
    mhz_pool = list(mhz_grid)
    if mhz_priority:
        for m in mhz_priority:
            if m in mhz_grid or m not in mhz_pool:
                mhz_pool.extend([m] * 2)
    mhz_pool = np.array(mhz_pool, dtype=np.float64)

    out: list[Candidate] = []
    for cid in range(num_candidates):
        mhz = float(rng.choice(mhz_pool))
        occ = random_occupancy(fixed_k, rng)
        out.append(
            Candidate(
                candidate_id=cid,
                occupancy=occ.astype(int).tolist(),
                mhz=mhz,
                k=fixed_k,
            )
        )
    return out
