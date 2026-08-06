"""Impedance spectrum loss for exp055 — lean stack, no redundant terms.

Terms:
  raw      — Huber on full spectrum, resonance-upweighted (freq_weight_alpha)
  deriv    — first-derivative (slope) match
  topk     — target top-k bin MSE with under-prediction penalty
  peak_idx — local peak index alignment (ramped via peak_scale)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def imp_ch0(x: torch.Tensor) -> torch.Tensor:
    return x[:, 0] if x.dim() == 3 else x


def resonance_freq_weights(target: torch.Tensor, alpha: float) -> torch.Tensor:
    if alpha <= 0 or target.shape[-1] < 3:
        return torch.ones_like(target)
    d2 = torch.diff(target, n=2, dim=-1)
    w = 1.0 + alpha * (-d2).clamp(min=0)
    pad = (w[..., :1] + w[..., -1:]) * 0.5
    w = torch.cat([pad[..., :1], w, pad[..., -1:]], dim=-1)
    return w / w.mean(dim=-1, keepdim=True).clamp(min=1e-6)


def local_peak_indices(x: torch.Tensor, num_peaks: int) -> torch.Tensor:
    b, f = x.shape
    if f < 3 or num_peaks <= 0:
        return torch.zeros(b, max(num_peaks, 1), dtype=torch.long, device=x.device)
    left, center, right = x[:, :-2], x[:, 1:-1], x[:, 2:]
    prom = (center - torch.maximum(left, right)) * ((center > left) & (center > right)).float()
    idx_1 = torch.arange(1, f - 1, device=x.device, dtype=torch.long).expand(b, -1)
    k = min(num_peaks, f - 2)
    _, order = prom.topk(k, dim=-1)
    peaks = idx_1.gather(1, order)
    if k < num_peaks:
        peaks = torch.cat([peaks, peaks[:, -1:].expand(b, num_peaks - k)], dim=1)
    return peaks


def _greedy_peak_index_loss(ti_n: torch.Tensor, ri_n: torch.Tensor, *, delta: float = 0.08) -> torch.Tensor:
    b, m = ti_n.shape
    if m <= 0:
        return ti_n.new_zeros(b)
    dist = (ri_n.unsqueeze(2) - ti_n.unsqueeze(1)).abs()
    used = torch.zeros(b, m, dtype=torch.bool, device=ti_n.device)
    total = ti_n.new_zeros(b)
    big = 1e4
    for j in range(m):
        d = dist[:, :, j].masked_fill(used, big)
        k = d.argmin(dim=1)
        used.scatter_(1, k.unsqueeze(1), True)
        ri_sel = ri_n.gather(1, k.unsqueeze(1)).squeeze(1)
        total = total + F.huber_loss(ri_sel, ti_n[:, j], delta=delta, reduction="none")
    return total / max(m, 1)


def peak_index_loss(recon: torch.Tensor, target: torch.Tensor, num_peaks: int) -> torch.Tensor:
    f = target.shape[-1]
    denom = max(f - 1, 1)
    m = min(num_peaks, max(f - 2, 1))
    ti = local_peak_indices(target, m)
    ri = local_peak_indices(recon, m)
    return _greedy_peak_index_loss(ti.float() / denom, ri.float() / denom, delta=0.08)


@dataclass
class ImpedanceSpectrumWeights:
    deriv_weight: float = 1.5
    topk_k: int = 20
    topk_weight: float = 7.0
    under_penalty: float = 2.8
    freq_weight_alpha: float = 2.0
    peak_index_weight: float = 2.0
    num_peaks: int = 8


def impedance_spectrum_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    *,
    imp_log_std: float = 1.0,
    penalty_scale: float = 1.0,
    peak_scale: float = 1.0,
    w: ImpedanceSpectrumWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if w is None:
        w = ImpedanceSpectrumWeights()

    s = imp_log_std
    r, t = imp_ch0(recon) * s, imp_ch0(target) * s
    raw_err = F.huber_loss(r, t, delta=1.0, reduction="none")
    if w.freq_weight_alpha > 0:
        fw = resonance_freq_weights(t, w.freq_weight_alpha)
        raw = (raw_err * fw).mean(1)
    else:
        raw = raw_err.mean(1)

    d1 = F.huber_loss(torch.diff(r, dim=-1), torch.diff(t, dim=-1), delta=1.0, reduction="none").mean(1)

    k = min(w.topk_k, t.shape[-1])
    idx = t.topk(k, dim=-1).indices
    err = (r.gather(1, idx) - t.gather(1, idx)).pow(2)
    under = (r.gather(1, idx) < t.gather(1, idx)).float()
    topk = (err * (1.0 + (w.under_penalty - 1.0) * under)).mean(1)

    base = raw + w.deriv_weight * d1 + penalty_scale * w.topk_weight * topk
    peak_extra = r.new_zeros(r.shape[0])
    if peak_scale > 0 and w.peak_index_weight > 0:
        peak_extra = penalty_scale * peak_scale * w.peak_index_weight * peak_index_loss(r, t, w.num_peaks)

    total = base + peak_extra
    return total, {"legacy": base, "peak_extra": peak_extra}
