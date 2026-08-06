"""Stratified MHz quotas for per-K AL candidate pools and ECAD selection."""
from __future__ import annotations

from typing import Any


def _parse_strata(raw: dict[str, Any] | dict[float, int]) -> dict[float, int]:
    return {float(k): int(v) for k, v in raw.items()}


def resolve_mhz_strata_per_k(cfg: dict[str, Any]) -> dict[float, int] | None:
    raw = cfg.get("mhz_strata_per_k")
    if not raw:
        return None
    return _parse_strata(raw)


def resolve_worst_mhz_strata_per_k(cfg: dict[str, Any]) -> dict[float, int] | None:
    """ECAD MHz quotas per K; auto-scaled from pool strata if omitted."""
    raw = cfg.get("worst_mhz_strata_per_k")
    if raw:
        return _parse_strata(raw)

    pool = resolve_mhz_strata_per_k(cfg)
    if not pool:
        return None

    from active_learning_pi.al.k_config import worst_per_k

    return scale_worst_mhz_strata_from_pool(pool, worst_per_k(cfg))


def scale_worst_mhz_strata_from_pool(pool: dict[float, int], n_worst: int) -> dict[float, int]:
    """Scale pool MHz strata to sum to ``n_worst`` using largest remainder."""
    n_worst = int(n_worst)
    if n_worst < 0:
        raise ValueError(f"n_worst must be >= 0, got {n_worst}")
    if n_worst == 0:
        return {}
    pool_total = sum(pool.values())
    quotas: dict[float, int] = {}
    remainders: list[tuple[float, float]] = []
    assigned = 0
    for mhz, count in sorted(pool.items()):
        exact = n_worst * count / pool_total
        base = int(exact)
        quotas[mhz] = base
        assigned += base
        remainders.append((exact - base, mhz))

    for _, mhz in sorted(remainders, reverse=True):
        if assigned >= n_worst:
            break
        quotas[mhz] = quotas.get(mhz, 0) + 1
        assigned += 1

    if assigned != n_worst:
        raise ValueError(f"Could not scale worst_mhz_strata to {n_worst} from {pool}")
    return quotas
