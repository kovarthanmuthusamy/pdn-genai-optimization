"""Impedance spectrum loss, in log-impedance units (normalized spectrum x ``imp_log_std``).

    total = raw + deriv_w * slope + penalty_scale * topk_w * topk          ("legacy" part)
          + penalty_scale * peak_scale * peak_w * peak_index               ("peak_extra" part)

  raw         Huber on the whole spectrum, resonances up-weighted (freq_weight_alpha)
  slope       Huber on the first difference
  topk        MSE on the target's top-k bins, under-prediction x under_penalty
  peak_index  greedy matching of the num_peaks most prominent local peaks (position only)

``penalty_scale`` and ``peak_scale`` are the curriculum ramps (training/schedule.py).

Usage (import only):
ImpedanceSpectrumWeights           the seven weights above, ``impedance_weights(c)`` builds it
impedance_spectrum_loss(recon, target, imp_log_std=, penalty_scale=, peak_scale=, w=)
                                   -> (B,) total, {"legacy": (B,), "peak_extra": (B,)}
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class ImpedanceSpectrumWeights:
    deriv_weight: float
    topk_k: int
    topk_weight: float
    under_penalty: float
    freq_weight_alpha: float
    peak_index_weight: float
    num_peaks: int


def impedance_weights(c) -> ImpedanceSpectrumWeights:
    return ImpedanceSpectrumWeights(
        deriv_weight=c.impedance_deriv_weight, topk_k=c.impedance_topk_k, topk_weight=c.impedance_topk_weight,
        under_penalty=c.impedance_under_penalty, freq_weight_alpha=c.impedance_freq_weight_alpha,
        peak_index_weight=c.impedance_peak_index_weight, num_peaks=c.impedance_num_peaks)


def _ch0(x: torch.Tensor) -> torch.Tensor:
    return x[:, 0] if x.dim() == 3 else x


def resonance_freq_weights(target: torch.Tensor, alpha: float) -> torch.Tensor:
    """Per-bin weights, larger where the target curves down (a resonance peak), mean 1."""
    d2 = torch.diff(target, n=2, dim=-1)
    w = 1.0 + alpha * (-d2).clamp(min=0)
    pad = (w[..., :1] + w[..., -1:]) * 0.5
    w = torch.cat([pad[..., :1], w, pad[..., -1:]], dim=-1)
    return w / w.mean(dim=-1, keepdim=True).clamp(min=1e-6)


def local_peak_indices(x: torch.Tensor, num_peaks: int) -> torch.Tensor:
    """(B, num_peaks) bin indices of the most prominent local maxima (last one repeated if fewer)."""
    b, f = x.shape
    if f < 3 or num_peaks <= 0:
        return torch.zeros(b, max(num_peaks, 1), dtype=torch.long, device=x.device)
    left, center, right = x[:, :-2], x[:, 1:-1], x[:, 2:]
    prom = (center - torch.maximum(left, right)) * ((center > left) & (center > right)).float()
    idx_1 = torch.arange(1, f - 1, device=x.device, dtype=torch.long).expand(b, -1)
    k = min(num_peaks, f - 2)
    peaks = idx_1.gather(1, prom.topk(k, dim=-1).indices)
    if k < num_peaks:
        peaks = torch.cat([peaks, peaks[:, -1:].expand(b, num_peaks - k)], dim=1)
    return peaks


def _greedy_peak_index_loss(ti_n: torch.Tensor, ri_n: torch.Tensor, *, delta: float = 0.08) -> torch.Tensor:
    """Each target peak takes the nearest unused predicted peak; Huber on the index distance."""
    b, m = ti_n.shape
    if m <= 0:
        return ti_n.new_zeros(b)
    dist = (ri_n.unsqueeze(2) - ti_n.unsqueeze(1)).abs()
    used = torch.zeros(b, m, dtype=torch.bool, device=ti_n.device)
    total = ti_n.new_zeros(b)
    for j in range(m):
        k = dist[:, :, j].masked_fill(used, 1e4).argmin(dim=1)
        used.scatter_(1, k.unsqueeze(1), True)
        total = total + F.huber_loss(ri_n.gather(1, k.unsqueeze(1)).squeeze(1), ti_n[:, j], delta=delta, reduction="none")
    return total / max(m, 1)


def peak_index_loss(recon: torch.Tensor, target: torch.Tensor, num_peaks: int) -> torch.Tensor:
    f = target.shape[-1]
    denom = max(f - 1, 1)
    m = min(num_peaks, max(f - 2, 1))
    ti, ri = local_peak_indices(target, m), local_peak_indices(recon, m)
    return _greedy_peak_index_loss(ti.float() / denom, ri.float() / denom, delta=0.08)


def impedance_spectrum_loss(recon: torch.Tensor, target: torch.Tensor, *, imp_log_std: float,
                            penalty_scale: float, peak_scale: float, w: ImpedanceSpectrumWeights,
                            ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    s = imp_log_std
    r, t = _ch0(recon) * s, _ch0(target) * s
    raw = (F.huber_loss(r, t, delta=1.0, reduction="none") * resonance_freq_weights(t, w.freq_weight_alpha)).mean(1)
    d1 = F.huber_loss(torch.diff(r, dim=-1), torch.diff(t, dim=-1), delta=1.0, reduction="none").mean(1)

    k = min(w.topk_k, t.shape[-1])
    idx = t.topk(k, dim=-1).indices
    err = (r.gather(1, idx) - t.gather(1, idx)).pow(2)
    under = (r.gather(1, idx) < t.gather(1, idx)).float()
    topk = (err * (1.0 + (w.under_penalty - 1.0) * under)).mean(1)

    base = raw + w.deriv_weight * d1 + penalty_scale * w.topk_weight * topk
    peak_extra = r.new_zeros(r.shape[0])
    if peak_scale > 0:                     # 0 before impedance_peak_start_epoch: skip the peak search
        peak_extra = penalty_scale * peak_scale * w.peak_index_weight * peak_index_loss(r, t, w.num_peaks)
    return base + peak_extra, {"legacy": base, "peak_extra": peak_extra}
