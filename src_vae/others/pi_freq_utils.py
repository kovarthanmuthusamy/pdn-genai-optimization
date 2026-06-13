"""PI frequency conditioning: MHz (user-facing) → model scalar in [0, 1].

Training data (dataloader) stores Hz on disk and yields log10-normalised tensors.
Inference / scripts pass frequency in **MHz**; use ``pi_freq_to_norm(..., unit='mhz')``.

Range matches multifreq training: 1 MHz … 600 MHz.
"""
from __future__ import annotations

import math
from typing import Literal, Union

import numpy as np
import torch

# Hz bounds (same as src_vae/others/dataloader.py)
PI_FREQ_MIN_HZ = 1e6
PI_FREQ_MAX_HZ = 600e6
_LOG10_MIN = math.log10(PI_FREQ_MIN_HZ)
_LOG10_RANGE = math.log10(PI_FREQ_MAX_HZ) - _LOG10_MIN

PiFreqUnit = Literal["mhz", "hz", "norm"]


def pi_freq_hz_to_norm(freq_hz: float) -> float:
    hz = max(float(freq_hz), 1.0)
    return float(np.clip((math.log10(hz) - _LOG10_MIN) / _LOG10_RANGE, 0.0, 1.0))


def pi_freq_mhz_to_norm(freq_mhz: float) -> float:
    """Convert PI frequency in MHz to model conditioning scalar in [0, 1]."""
    return pi_freq_hz_to_norm(float(freq_mhz) * 1e6)


def pi_freq_to_norm(
    freq: Union[float, int, np.ndarray, torch.Tensor],
    *,
    unit: PiFreqUnit = "mhz",
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert PI frequency to (N,) normalised tensor for the VAE / physics critic.

    Args:
        freq: Scalar or 1-D values. Interpretation depends on ``unit``:
            - ``mhz``: frequency in MHz (default for inference APIs)
            - ``hz``: frequency in Hz (dataset files on disk)
            - ``norm``: already log10-normalised in [0, 1] (training batches)
    """
    if unit == "norm":
        t = torch.as_tensor(freq, dtype=dtype, device=device).float().flatten()
        return t.clamp(0.0, 1.0)

    if isinstance(freq, torch.Tensor):
        arr = freq.detach().cpu().numpy().astype(np.float64).flatten()
    else:
        arr = np.asarray(freq, dtype=np.float64).flatten()

    if unit == "mhz":
        norms = [pi_freq_mhz_to_norm(x) for x in arr]
    else:
        norms = [pi_freq_hz_to_norm(x) for x in arr]

    return torch.tensor(norms, dtype=dtype, device=device)


def pi_freq_norm_for_model(
    freq: Union[float, int, np.ndarray, torch.Tensor] | None,
    batch_size: int,
    *,
    unit: PiFreqUnit = "mhz",
    default_mhz: float = 200.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build (B,) PI_freq tensor; default operating point if ``freq`` is None."""
    if freq is None:
        v = pi_freq_mhz_to_norm(default_mhz)
        return torch.full((batch_size,), v, device=device, dtype=dtype)
    t = pi_freq_to_norm(freq, unit=unit, device=device, dtype=dtype)
    if t.numel() == 1 and batch_size > 1:
        t = t.expand(batch_size)
    elif t.shape[0] != batch_size:
        raise ValueError(f"PI_freq length {t.shape[0]} != batch_size {batch_size}")
    return t
