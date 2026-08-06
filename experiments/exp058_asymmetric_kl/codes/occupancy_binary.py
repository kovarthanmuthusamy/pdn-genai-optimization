"""Binary occupancy for heatmap decode — matches CAD / ECADStar discrete layouts.

Fully vectorized (no Python loops / no GPU->CPU syncs) so it is safe to call on the
training decode hot path. On already-binary GT occupancy it is a no-op (returns the
same 0/1 grid with exactly K ones).
"""

from __future__ import annotations

import torch


def topk_occ_binary(occ_prob: torch.Tensor, k: "int | torch.Tensor") -> torch.Tensor:
    """Hard top-K mask on 52-d occupancy (B, 52). Matches latent-opt / PEB binarization.

    Vectorized: per-row variable K handled via a descending-sort threshold, so there is
    no Python loop and no per-element ``.item()`` sync. For binary GT (values in {0,1})
    the k-th largest value is exactly 1.0, so the result has exactly K ones and matches
    the reference top-K implementation.
    """
    if occ_prob.dim() == 1:
        occ_prob = occ_prob.unsqueeze(0)
    _, n = occ_prob.shape

    if isinstance(k, torch.Tensor):
        k_per = k.reshape(-1).to(occ_prob.device).long().clamp(0, n)          # (B,)
        sorted_vals, _ = occ_prob.sort(dim=-1, descending=True)               # (B, N)
        thr_idx = (k_per - 1).clamp(min=0)                                    # (B,)
        thr = sorted_vals.gather(1, thr_idx.unsqueeze(1))                     # (B, 1)
        out = (occ_prob >= thr).float()
        return out.masked_fill((k_per == 0).unsqueeze(1), 0.0)

    ki = max(0, min(int(k), n))
    if ki <= 0:
        return torch.zeros_like(occ_prob)
    idx = occ_prob.topk(ki, dim=-1).indices
    return torch.zeros_like(occ_prob).scatter(-1, idx, 1.0)


def is_binary_occupancy(occ: torch.Tensor, *, atol: float = 1e-4) -> bool:
    """Cheap CPU-syncing check — for diagnostics only, NOT the decode hot path."""
    if occ.numel() == 0:
        return True
    return bool(torch.all((occ <= atol) | (occ >= 1.0 - atol)))


def occupancy_for_heatmap_decode(
    occ: "torch.Tensor | None",
    k: "int | torch.Tensor",
    *,
    force_binary: bool = True,
    ste: bool = False,
) -> "torch.Tensor | None":
    """Binarize occupancy before decoder spatial conditioning (CAD-aligned).

    Vectorized and sync-free. On binary GT this returns an equivalent 0/1 grid with
    exactly K ones (no-op in effect). When ``ste=True`` and ``occ`` requires grad, the
    forward value is hard top-K but gradients flow through the soft ``occ``.
    """
    if occ is None or not force_binary:
        return occ
    hard = topk_occ_binary(occ, k)
    if ste and occ.requires_grad:
        return hard + (occ - occ.detach())
    return hard
