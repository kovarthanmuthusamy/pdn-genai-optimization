"""Peak-aware impedance losses (frequency weighting, dual top-k, peak alignment)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def imp_ch0(x: torch.Tensor) -> torch.Tensor:
    return x[:, 0] if x.dim() == 3 else x


def resonance_freq_weights(target: torch.Tensor, alpha: float) -> torch.Tensor:
    """Up-weight bins where target log-Z has strong negative curvature (resonances)."""
    if alpha <= 0 or target.shape[-1] < 3:
        return torch.ones_like(target)
    d2 = torch.diff(target, n=2, dim=-1)
    w = 1.0 + alpha * (-d2).clamp(min=0)
    pad = (w[..., :1] + w[..., -1:]) * 0.5
    w = torch.cat([pad[..., :1], w, pad[..., -1:]], dim=-1)
    return w / w.mean(dim=-1, keepdim=True).clamp(min=1e-6)


def local_peak_indices(x: torch.Tensor, num_peaks: int) -> torch.Tensor:
    """Strongest local maxima indices per row, shape (B, num_peaks)."""
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


def dual_topk_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    k: int,
    under_penalty: float = 1.0,
) -> torch.Tensor:
    """Top-k on target bins plus top-k on recon bins (catches misplaced peaks)."""
    f = target.shape[-1]
    k = min(k, f)
    if k <= 0:
        return recon.new_zeros(recon.shape[0])

    idx_t = target.topk(k, dim=-1).indices
    idx_r = recon.topk(k, dim=-1).indices
    err_t = (recon.gather(1, idx_t) - target.gather(1, idx_t)).pow(2)
    err_r = (recon.gather(1, idx_r) - target.gather(1, idx_r)).pow(2)
    if under_penalty > 1.0:
        err_t = err_t * (1.0 + (under_penalty - 1.0) * (recon.gather(1, idx_t) < target.gather(1, idx_t)).float())
        err_r = err_r * (1.0 + (under_penalty - 1.0) * (recon.gather(1, idx_r) < target.gather(1, idx_r)).float())
    return 0.5 * (err_t.mean(1) + err_r.mean(1))


def _greedy_peak_index_loss(
    ti_n: torch.Tensor,
    ri_n: torch.Tensor,
    *,
    delta: float = 0.08,
) -> torch.Tensor:
    """Match each target peak to nearest unused recon peak (normalized frequency)."""
    b, m = ti_n.shape
    out = ti_n.new_zeros(b)
    for bi in range(b):
        used = torch.zeros(m, dtype=torch.bool, device=ti_n.device)
        acc = ti_n.new_zeros(())
        for j in range(m):
            dist = (ri_n[bi] - ti_n[bi, j]).abs().masked_fill(used, 1e4)
            k = int(dist.argmin().item())
            used[k] = True
            acc = acc + F.huber_loss(ri_n[bi, k], ti_n[bi, j], delta=delta, reduction="sum")
        out[bi] = acc / max(m, 1)
    return out


def peak_alignment_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    num_peaks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize peak position drift (greedy match) and magnitude at target peaks."""
    f = target.shape[-1]
    denom = max(f - 1, 1)
    m = min(num_peaks, max(f - 2, 1))
    ti = local_peak_indices(target, m)
    ri = local_peak_indices(recon, m)
    ti_n = ti.float() / denom
    ri_n = ri.float() / denom
    idx_loss = _greedy_peak_index_loss(ti_n, ri_n, delta=0.08)
    mag_loss = F.huber_loss(
        recon.gather(1, ti), target.gather(1, ti), delta=1.0, reduction="none",
    ).mean(1)
    return idx_loss, mag_loss


@dataclass
class ImpedanceSpectrumWeights:
    deriv_weight: float = 1.5
    concavity_weight: float = 2.5
    topk_k: int = 20
    topk_weight: float = 7.0
    under_penalty: float = 2.8
    freq_weight_alpha: float = 2.0
    dual_topk_weight: float = 1.0
    peak_index_weight: float = 2.0
    peak_mag_weight: float = 1.5
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
    """Per-sample impedance loss (B,) and components for logging."""
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

    d2 = torch.diff(t, n=2, dim=-1)
    cw = (-d2).clamp(min=0) / ((-d2).clamp(min=0).mean(1, keepdim=True) + 1e-6)
    conc = (raw_err[:, 1:-1] * cw).mean(1)
    d1 = F.huber_loss(torch.diff(r, dim=-1), torch.diff(t, dim=-1), delta=1.0, reduction="none").mean(1)

    k = min(w.topk_k, t.shape[-1])
    idx = t.topk(k, dim=-1).indices
    err = (r.gather(1, idx) - t.gather(1, idx)).pow(2)
    under = (r.gather(1, idx) < t.gather(1, idx)).float()
    topk = (err * (1.0 + (w.under_penalty - 1.0) * under)).mean(1)

    extra = r.new_zeros(r.shape[0])
    if w.dual_topk_weight > 0 and peak_scale > 0:
        extra = extra + w.dual_topk_weight * dual_topk_loss(r, t, k, w.under_penalty)
    if peak_scale > 0 and (w.peak_index_weight > 0 or w.peak_mag_weight > 0):
        p_idx, p_mag = peak_alignment_loss(r, t, w.num_peaks)
        extra = extra + w.peak_index_weight * p_idx + w.peak_mag_weight * p_mag

    legacy = raw + w.deriv_weight * d1 + penalty_scale * (w.concavity_weight * conc + w.topk_weight * topk)
    peak_extra = penalty_scale * peak_scale * extra
    total = legacy + peak_extra
    return total, {"legacy": legacy, "peak_extra": peak_extra}


def surrogate_spectrum_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    ch0_weight: float = 1.0,
    topk_k: int = 20,
    topk_weight: float = 7.0,
    under_penalty: float = 2.8,
    freq_weight_alpha: float = 2.0,
    dual_topk_weight: float = 1.0,
    peak_index_weight: float = 2.0,
    peak_mag_weight: float = 1.5,
    num_peaks: int = 8,
    peak_scale: float = 1.0,
) -> torch.Tensor:
    """Scalar loss for occ → impedance surrogate."""
    w = ImpedanceSpectrumWeights(
        deriv_weight=0.0,
        concavity_weight=0.0,
        topk_k=topk_k,
        topk_weight=topk_weight,
        under_penalty=under_penalty,
        freq_weight_alpha=freq_weight_alpha,
        dual_topk_weight=dual_topk_weight,
        peak_index_weight=peak_index_weight,
        peak_mag_weight=peak_mag_weight,
        num_peaks=num_peaks,
    )
    per, _ = impedance_spectrum_loss(
        pred, target, imp_log_std=1.0, penalty_scale=1.0, peak_scale=peak_scale, w=w,
    )
    if ch0_weight != 1.0:
        per = ch0_weight * per
    return per.mean()
