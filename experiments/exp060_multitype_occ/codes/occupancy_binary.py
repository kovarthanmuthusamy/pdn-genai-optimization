"""Occupancy harden helpers (legacy binary + multi-type one-hot).

Heatmap spatial paths use per-slot presence. Multi-type layouts use
``occupancy_types`` one-hot (B, 52, C); this module keeps the binary top-K
utilities and a compatible ``occupancy_for_heatmap_decode`` entry point.
"""
from __future__ import annotations

import torch

from experiments.exp060_multitype_occ.codes.occupancy_types import (
    occupancy_for_heatmap_decode as _occ_hm_decode_typed,
)


def topk_occ_binary(occ_prob: torch.Tensor, k: "int | torch.Tensor") -> torch.Tensor:
    """Hard top-K mask on 52-d occupancy (B, 52)."""
    if occ_prob.dim() == 1:
        occ_prob = occ_prob.unsqueeze(0)
    if occ_prob.dim() == 3:
        # presence from one-hot/probs
        from experiments.exp060_multitype_occ.codes.occupancy_types import presence_from_onehot
        occ_prob = presence_from_onehot(occ_prob)
    _, n = occ_prob.shape

    if isinstance(k, torch.Tensor):
        k_per = k.reshape(-1).to(occ_prob.device).long().clamp(0, n)
        sorted_vals, _ = occ_prob.sort(dim=-1, descending=True)
        thr_idx = (k_per - 1).clamp(min=0)
        thr = sorted_vals.gather(1, thr_idx.unsqueeze(1))
        out = (occ_prob >= thr).float()
        return out.masked_fill((k_per == 0).unsqueeze(1), 0.0)

    ki = max(0, min(int(k), n))
    if ki <= 0:
        return torch.zeros_like(occ_prob)
    idx = occ_prob.topk(ki, dim=-1).indices
    return torch.zeros_like(occ_prob).scatter(-1, idx, 1.0)


def is_binary_occupancy(occ: torch.Tensor, *, atol: float = 1e-4) -> bool:
    if occ.numel() == 0:
        return True
    if occ.dim() == 3:
        # one-hot rows
        return bool(torch.all((occ.sum(dim=-1) - 1.0).abs() <= atol))
    return bool(torch.all((occ <= atol) | (occ >= 1.0 - atol)))


def occupancy_for_heatmap_decode(
    occ: "torch.Tensor | None",
    k: "int | torch.Tensor",
    *,
    force_binary: bool = True,
    force_hard: bool | None = None,
    ste: bool = False,
) -> "torch.Tensor | None":
    """Return presence (B, 52) for heatmap spatial conditioning."""
    hard = force_binary if force_hard is None else force_hard
    return _occ_hm_decode_typed(occ, k, force_hard=hard, ste=ste)
