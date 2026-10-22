"""Placement formats: slot labels and grid positions, type codes, one-hot tensors, top-K hardening.

What: a placement is 52 slots, each empty or holding one of T decap parts (T = 3:
    T1 = 100 nF, T2 = 47 nF, T3 = 22 nF).

        on disk     type.npy   (52,) integer codes, 0 = empty, 1..T = part
        in a batch  one-hot    (B, 52, T) float, empty slot = all-zero row

    "Hardening" turns per-slot scores into the placement CAD would build: the K highest-scoring
    slots, each with its most likely part.

Usage (import only):
    LABELS_ORDERED, LABEL_TO_COORD      slot label per index; label -> (row, col) on the 7 x 8 grid
    type_ids_to_onehot(codes, T)        codes (52,) / (B,52), or 0/1 presence -> (..., 52, T)
    ensure_batch_onehot(occ, T)         any accepted batch form -> (B, 52, T) float
    presence_from_onehot(occ)           (..., 52, T) -> (..., 52) occupied mask
    hard_onehot_from_logits(logits, K, ste=False)  per-slot part logits (B,52,T) -> hard one-hot with
                                        K decaps; with ``ste`` the gradient passes straight through
    hard_onehot_from_probs(occ, K)      the same from probabilities / one-hot (logits detected by range)
    topk_presence(presence, K)          (B,52) scores -> hard 0/1 mask with K ones
    hard_presence(occ, K, ste=False)    hard (B,52) presence from a one-hot or presence input; with
                                        ``ste`` the gradient passes straight through the hardening
"""
from __future__ import annotations

from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

NUM_SLOTS = 52
IntTensor = Union[torch.Tensor, np.ndarray]

# Slot labels in vector order: index 0 is slot "C1", index 51 is slot "C52".
LABELS_ORDERED = [f"C{i}" for i in range(1, NUM_SLOTS + 1)]


def _grid_positions() -> dict:
    """(row, col) -> label on the 7 x 8 slot grid. The four cells (0,3), (0,4), (6,3), (6,4) hold no
    slot; the first row reads C4 C5 C6 . . C1 C2 C3, every further row continues from C7 in order."""
    pos = {(0, 0): "C4", (0, 1): "C5", (0, 2): "C6", (0, 5): "C1", (0, 6): "C2", (0, 7): "C3"}
    no_slot = {(0, 3), (0, 4), (6, 3), (6, 4)}
    nxt = 7
    for row in range(1, 7):
        for col in range(8):
            if (row, col) not in no_slot:
                pos[(row, col)] = f"C{nxt}"
                nxt += 1
    return pos


LABEL_TO_COORD = {label: coord for coord, label in _grid_positions().items()}


def _codes_to_onehot(codes: torch.Tensor, n_types: int, *, dtype: torch.dtype) -> torch.Tensor:
    """Codes in 0..T -> (..., T) one-hot; 0 becomes the all-zero row."""
    oh = F.one_hot(codes.clamp(0, n_types), num_classes=n_types + 1).to(dtype=dtype)
    return oh[..., 1:]


def type_ids_to_onehot(type_ids: IntTensor, n_decap_types: int, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Type codes -> (..., 52, T) one-hot; empty slots stay all-zero.

    Accepts (52,) or (B, 52) integer codes (0/1 presence reads as empty / type 1), or an
    already one-hot (52, T) / (B, 52, T) tensor.
    """
    t = int(n_decap_types)
    x = torch.from_numpy(np.asarray(type_ids)) if isinstance(type_ids, np.ndarray) else type_ids
    if x.dim() == 1 and x.numel() == NUM_SLOTS:
        return _codes_to_onehot(x.long(), t, dtype=dtype)
    if x.dim() == 2 and x.shape == (NUM_SLOTS, t):
        return x.to(dtype=dtype)
    if x.dim() == 2 and x.shape[-1] == NUM_SLOTS:
        return _codes_to_onehot(x.long(), t, dtype=dtype)
    if x.dim() == 3 and x.shape[1:] == (NUM_SLOTS, t):
        return x.to(dtype=dtype)
    raise ValueError(f"expected codes (52,) / (B,52) or one-hot (52,{t}) / (B,52,{t}); got {tuple(x.shape)}")


def presence_from_onehot(occ: torch.Tensor) -> torch.Tensor:
    """Occupied mask (..., 52): any part channel on."""
    return occ.sum(dim=-1).clamp(0.0, 1.0)


def ensure_batch_onehot(occ: torch.Tensor, n_decap_types: int) -> torch.Tensor:
    """Batch occupancy in any accepted form -> (B, 52, T) float one-hot."""
    t = int(n_decap_types)
    if occ.dim() == 3 and occ.shape[1:] == (NUM_SLOTS, t):
        return occ.float()
    if occ.dim() == 2 and occ.shape == (NUM_SLOTS, t):
        return occ.unsqueeze(0).float()
    return type_ids_to_onehot(occ, t)


def _per_sample_k(k: "int | torch.Tensor", b: int, n: int, device) -> torch.Tensor:
    if isinstance(k, torch.Tensor):
        return k.reshape(-1).to(device).long().clamp(0, n)
    return torch.full((b,), max(0, min(int(k), n)), device=device, dtype=torch.long)


def hard_onehot_from_logits(logits: torch.Tensor, k: "int | torch.Tensor", *, ste: bool = False) -> torch.Tensor:
    """(B, 52, T) part logits -> hard one-hot: the K slots with the highest max-part probability,
    each set to its argmax part; every other slot empty (the search's read-out). With ``ste`` and logits that require grad, the forward value is the hard
    one-hot and the gradient goes to the part probabilities as if there were no hardening."""
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    if logits.dim() != 3:
        raise ValueError(f"expected (B, 52, T) logits, got {tuple(logits.shape)}")
    b, n, t = logits.shape
    score = torch.sigmoid(logits).max(dim=-1).values
    k_per = _per_sample_k(k, b, n, logits.device)
    thr = score.sort(dim=-1, descending=True).values.gather(1, (k_per - 1).clamp(min=0).unsqueeze(1))
    keep = (score >= thr) & (k_per > 0).unsqueeze(1)
    oh = F.one_hot(logits.argmax(dim=-1), num_classes=t).to(dtype=logits.dtype)
    hard = oh * keep.unsqueeze(-1).to(dtype=logits.dtype)
    if ste and logits.requires_grad:
        probs = torch.sigmoid(logits)
        hard = hard + (probs - probs.detach())
    return hard


def hard_onehot_from_probs(occ: torch.Tensor, k: "int | torch.Tensor") -> torch.Tensor:
    """Harden (B, 52, T) probabilities or a one-hot (values outside [0, 1] are read as logits)."""
    if occ.dim() == 2 and occ.shape[0] == NUM_SLOTS:
        occ = occ.unsqueeze(0)
    if occ.dim() != 3:
        raise ValueError(f"expected (B, 52, T), got {tuple(occ.shape)}")
    if float(occ.min()) < -1e-3 or float(occ.max()) > 1.0 + 1e-3:
        return hard_onehot_from_logits(occ, k)
    probs = occ.clamp(1e-6, 1.0 - 1e-6)
    return hard_onehot_from_logits(torch.log(probs) - torch.log1p(-probs), k)


def topk_presence(presence: torch.Tensor, k: "int | torch.Tensor") -> torch.Tensor:
    """(B, 52) scores -> 0/1 mask of the K highest (ties at the K-th value are all kept)."""
    if presence.dim() == 1:
        presence = presence.unsqueeze(0)
    b, n = presence.shape
    k_per = _per_sample_k(k, b, n, presence.device)
    thr = presence.sort(dim=-1, descending=True).values.gather(1, (k_per - 1).clamp(min=0).unsqueeze(1))
    return (presence >= thr).float().masked_fill((k_per == 0).unsqueeze(1), 0.0)


def hard_presence(occ: torch.Tensor, k: "int | torch.Tensor", *, ste: bool = False) -> torch.Tensor:
    """Hard (B, 52) presence with exactly K decaps, from (B, 52, T) or (B, 52) scores. With ``ste``
    and an input that requires grad, the gradient passes straight through the hardening."""
    if occ.dim() == 3:
        hard = hard_onehot_from_probs(occ, k)
        if ste and occ.requires_grad:
            hard = hard + (occ - occ.detach())
        return presence_from_onehot(hard)
    hard = topk_presence(occ, k)
    if ste and occ.requires_grad:
        hard = hard + (occ - occ.detach())
    return hard
