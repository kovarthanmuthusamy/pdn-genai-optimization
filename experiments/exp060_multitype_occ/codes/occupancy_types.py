"""Multi-type decap occupancy: one-hot over catalog types only (no empty channel).

Canonical layout tensor for exp060:

    occ ∈ {0,1}^{N × T} with N = 52
    empty slot     = all zeros
    type t (1..T)  = one-hot on channel (t-1)

On disk prefer ``type.npy`` with integer codes in {0,...,T} (0 = empty).
Legacy binary ``occ.npy`` (0/1) maps to empty / type-1.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

NUM_SLOTS = 52

IntTensor = Union[torch.Tensor, np.ndarray]


def n_occ_classes(n_decap_types: int) -> int:
    """Number of type channels (= T). Empty is the zero vector, not a channel."""
    t = int(n_decap_types)
    if t < 1:
        raise ValueError(f"n_decap_types must be >= 1, got {t}")
    return t


def type_ids_to_onehot(
    type_ids: IntTensor,
    n_decap_types: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert type codes → (..., N, T) one-hot; empty codes stay all-zero rows.

    Accepts:
      - (N,) or (B, N) integer codes in {0,...,T}
      - (N,) or (B, N) binary occupancy (0/1) → empty / type-1
      - (N, T) or (B, N, T) already one-hot
    """
    t = n_occ_classes(n_decap_types)
    if isinstance(type_ids, np.ndarray):
        x = torch.from_numpy(np.asarray(type_ids))
    else:
        x = type_ids

    if x.dim() == 1:
        if x.numel() != NUM_SLOTS:
            raise ValueError(f"expected length-{NUM_SLOTS} type vector, got shape {tuple(x.shape)}")
        return _codes_to_onehot(x.long(), t, dtype=dtype)

    if x.dim() == 2:
        if x.shape[-1] == t and x.shape[0] == NUM_SLOTS:
            return x.to(dtype=dtype)
        if x.shape[-1] == NUM_SLOTS:
            return _codes_to_onehot(x.long(), t, dtype=dtype)
        raise ValueError(
            f"expected (B,{NUM_SLOTS}) codes or ({NUM_SLOTS},{t}) one-hot, got {tuple(x.shape)}"
        )

    if x.dim() == 3:
        if x.shape[1] == NUM_SLOTS and x.shape[2] == t:
            return x.to(dtype=dtype)
        raise ValueError(f"expected (B,{NUM_SLOTS},{t}) one-hot, got {tuple(x.shape)}")

    raise ValueError(f"unsupported occupancy rank {x.dim()} shape {tuple(x.shape)}")


def _codes_to_onehot(codes: torch.Tensor, n_types: int, *, dtype: torch.dtype) -> torch.Tensor:
    """codes in {0,...,T}; 0 → zero row; k≥1 → channel k-1."""
    codes = codes.clamp(0, n_types)
    oh = F.one_hot(codes, num_classes=n_types + 1).to(dtype=dtype)  # includes empty ch0
    return oh[..., 1:]  # drop empty channel → (..., T)


def onehot_to_type_ids(occ: torch.Tensor) -> torch.Tensor:
    """(B, N, T) or (N, T) → integer codes (..., N); empty rows → 0."""
    if occ.dim() < 2:
        raise ValueError(f"expected one-hot with type dim, got {tuple(occ.shape)}")
    pres = presence_from_onehot(occ)
    typ = occ.argmax(dim=-1) + 1  # 1..T
    return torch.where(pres > 0.5, typ, torch.zeros_like(typ))


def presence_from_onehot(occ: torch.Tensor) -> torch.Tensor:
    """Occupied mask: any type channel on. Shape (..., N)."""
    if occ.dim() >= 2 and occ.shape[-1] >= 1:
        return occ.sum(dim=-1).clamp(0.0, 1.0)
    return occ.clamp(0.0, 1.0)


def k_from_onehot(occ: torch.Tensor) -> torch.Tensor:
    """K = number of non-empty slots."""
    pres = presence_from_onehot(occ)
    if pres.dim() == 1:
        return pres.sum()
    return pres.sum(dim=-1)


def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """(B, N, T) logits → independent sigmoid type probs (empty ≈ all low)."""
    return torch.sigmoid(logits)


def hard_onehot_from_logits(
    logits: torch.Tensor,
    k: "int | torch.Tensor",
) -> torch.Tensor:
    """CAD-aligned hard layout from per-slot type logits (B, N, T).

    1. Score slots by max type probability.
    2. Keep top-K slots.
    3. On kept slots: one-hot argmax type.
    4. Elsewhere: all zeros (empty).
    """
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    if logits.dim() != 3:
        raise ValueError(f"expected (B, N, T) logits, got {tuple(logits.shape)}")
    b, n, t = logits.shape
    if t < 1:
        raise ValueError("need at least one type channel")

    probs = logits_to_probs(logits)
    score = probs.max(dim=-1).values  # (B, N)

    if isinstance(k, torch.Tensor):
        k_per = k.reshape(-1).to(logits.device).long().clamp(0, n)
    else:
        k_per = torch.full((b,), max(0, min(int(k), n)), device=logits.device, dtype=torch.long)

    sorted_vals, _ = score.sort(dim=-1, descending=True)
    thr_idx = (k_per - 1).clamp(min=0)
    thr = sorted_vals.gather(1, thr_idx.unsqueeze(1))
    keep = (score >= thr) & (k_per > 0).unsqueeze(1)

    type_id = logits.argmax(dim=-1)  # 0..T-1
    oh = F.one_hot(type_id, num_classes=t).to(dtype=logits.dtype)
    return oh * keep.unsqueeze(-1).to(dtype=logits.dtype)


def hard_onehot_from_probs(
    occ: torch.Tensor,
    k: "int | torch.Tensor",
) -> torch.Tensor:
    """Harden (B, N, T) one-hot/probs/logits with budget K."""
    if occ.dim() == 2 and occ.shape[0] == NUM_SLOTS:
        occ = occ.unsqueeze(0)
    if occ.dim() != 3:
        raise ValueError(f"expected (B, N, T), got {tuple(occ.shape)}")
    # Heuristic: values outside [0,1] or row sums >> 1 → treat as logits
    if float(occ.min()) < -1e-3 or float(occ.max()) > 1.0 + 1e-3:
        return hard_onehot_from_logits(occ, k)
    # probs / one-hot: convert to logits via logit for shared path
    probs = occ.clamp(1e-6, 1.0 - 1e-6)
    logits = torch.log(probs) - torch.log1p(-probs)
    return hard_onehot_from_logits(logits, k)


def occupancy_for_heatmap_decode(
    occ: "torch.Tensor | None",
    k: "int | torch.Tensor",
    *,
    force_hard: bool = True,
    ste: bool = False,
) -> "torch.Tensor | None":
    """Return presence (B, N) for heatmap spatial grids after optional hard budget."""
    if occ is None:
        return None
    if not force_hard:
        if occ.dim() == 3:
            return presence_from_onehot(occ)
        return occ

    if occ.dim() == 3:
        hard = hard_onehot_from_probs(occ, k)
        if ste and occ.requires_grad:
            hard = hard + (occ - occ.detach())
        return presence_from_onehot(hard)

    from experiments.exp060_multitype_occ.codes.occupancy_binary import topk_occ_binary

    hard = topk_occ_binary(occ, k)
    if ste and occ.requires_grad:
        return hard + (occ - occ.detach())
    return hard


def ensure_batch_onehot(occ: torch.Tensor, n_decap_types: int) -> torch.Tensor:
    """Normalize batch occupancy to (B, N, T) float one-hot."""
    t = n_occ_classes(n_decap_types)
    if occ.dim() == 3 and occ.shape[1] == NUM_SLOTS and occ.shape[2] == t:
        return occ.float()
    if occ.dim() == 2 and occ.shape[-1] == NUM_SLOTS:
        return type_ids_to_onehot(occ, n_decap_types)
    if occ.dim() == 2 and occ.shape[0] == NUM_SLOTS and occ.shape[1] == t:
        return occ.unsqueeze(0).float()
    return type_ids_to_onehot(occ, n_decap_types)
