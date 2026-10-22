"""Board region of the 64 x 64 map grid, as geometry.

What: the board is the fixed region in inputs/binary_mask.npy (83.79 % of the grid). Every
    "on the board" decision in training, validation and evaluation uses it, never a value threshold:
    maps are normalized per frequency, so the background value differs at every frequency.
Usage: board_mask(device, dtype) -> (1, 1, 64, 64) mask, True / 1 on the board (cached);
    board_fg(target) -> float mask with target's shape; zero_offboard(hm) -> hm with off-board pixels 0.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

INPUTS = Path(__file__).resolve().parent / "inputs"
_CACHE: dict[tuple, torch.Tensor] = {}


def board_mask(device: torch.device | str, dtype: torch.dtype = torch.bool) -> torch.Tensor:
    """(1, 1, 64, 64) board mask, cached per device and dtype. True = on the board."""
    key = (str(device), dtype)
    if key not in _CACHE:
        arr = np.load(INPUTS / "binary_mask.npy")
        _CACHE[key] = torch.as_tensor(arr, dtype=dtype, device=device).reshape(1, 1, *arr.shape)
    return _CACHE[key]


def board_fg(target: torch.Tensor) -> torch.Tensor:
    """Float foreground mask with ``target``'s shape ((B,1,H,W) or (B,H,W))."""
    m = board_mask(target.device, torch.float32)
    if target.dim() == 3:
        m = m[:, 0]
    return m.expand_as(target)


def zero_offboard(hm: torch.Tensor) -> torch.Tensor:
    """Set everything outside the board to 0, so the background carries no signal."""
    m = board_mask(hm.device, torch.bool)
    if hm.dim() == 3:
        m = m[:, 0]
    return hm * m.expand_as(hm)
