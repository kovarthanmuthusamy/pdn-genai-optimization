"""Resolve decap-count K settings for active-learning candidate pools."""
from __future__ import annotations

from typing import Any


def resolve_k_values(cfg: dict[str, Any]) -> list[int]:
    """Return sorted K list from ``k_values``, ``k_min``/``k_max``, or ``fixed_k``."""
    if cfg.get("k_values"):
        return sorted({int(k) for k in cfg["k_values"]})
    k_min, k_max = cfg.get("k_min"), cfg.get("k_max")
    if k_min is not None and k_max is not None:
        lo, hi = int(k_min), int(k_max)
        if lo > hi:
            raise ValueError(f"k_min ({lo}) must be <= k_max ({hi})")
        return list(range(lo, hi + 1))
    return [int(cfg.get("fixed_k", 5))]


def use_per_k_pools(cfg: dict[str, Any]) -> bool:
    """When true, each K gets its own candidate pool (e.g. 400) and worst-N selection."""
    if "k_sweep_per_pool" in cfg:
        return bool(cfg["k_sweep_per_pool"])
    return cfg.get("k_min") is not None and cfg.get("k_max") is not None


def candidates_per_k(cfg: dict[str, Any]) -> int:
    return int(cfg.get("candidates_per_k") or cfg.get("num_candidates", 400))


def worst_per_k(cfg: dict[str, Any]) -> int:
    return int(cfg.get("worst_per_k") or cfg.get("simulate_batch_size", 10))


def resolve_worst_per_k_values(cfg: dict[str, Any]) -> dict[int, int] | None:
    """
    Optional: distribute a total ECAD target across K values.

    If ``total_ecad_per_cycle`` is set and per-K pools are enabled, we assign either
    ⌊total/K⌋ or ⌈total/K⌉ per K (first Ks get the +1 remainder) so totals match exactly.
    """
    if not use_per_k_pools(cfg):
        return None
    total = cfg.get("total_ecad_per_cycle")
    if total is None:
        return None
    total = int(total)
    ks = resolve_k_values(cfg)
    if total < 0:
        raise ValueError(f"total_ecad_per_cycle must be >= 0, got {total}")
    if not ks:
        return None
    base = total // len(ks)
    rem = total % len(ks)
    out: dict[int, int] = {}
    for i, k in enumerate(ks):
        out[int(k)] = int(base + (1 if i < rem else 0))
    return out


def total_ecad_batch_size(cfg: dict[str, Any]) -> int:
    if use_per_k_pools(cfg):
        dist = resolve_worst_per_k_values(cfg)
        if dist:
            return int(sum(dist.values()))
        return len(resolve_k_values(cfg)) * worst_per_k(cfg)
    return int(cfg.get("simulate_batch_size") or cfg.get("batch_size", 8))


def stratify_ecad_by_k(cfg: dict[str, Any]) -> bool:
    if use_per_k_pools(cfg):
        return False
    acq = cfg.get("acquisition") or {}
    if "stratify_by_k" in acq:
        return bool(acq["stratify_by_k"])
    return bool(cfg.get("stratify_ecad_by_k", True))
