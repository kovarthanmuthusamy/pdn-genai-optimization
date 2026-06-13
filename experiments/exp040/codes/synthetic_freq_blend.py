"""Synthetic between-anchor heatmap targets for multifreq training (exp040)."""

from __future__ import annotations

import torch


def maybe_apply_synthetic_blend(batch: dict, cfg) -> dict:
    """
    With probability ``synthetic_blend_prob``, replace (heatmap, PI_freq) by a
    convex blend of native and alt anchor maps from the same layout.

    Teaches decode at off-anchor MHz without new ECAD runs.
    """
    prob = float(getattr(cfg, "synthetic_blend_prob", 0.0))
    if prob <= 0.0:
        return batch
    if "heatmap_norm_alt" not in batch or "PI_freq_alt" not in batch:
        return batch
    if not hasattr(batch["heatmap_norm"], "device"):
        return batch

    device = batch["heatmap_norm"].device
    B = batch["heatmap_norm"].shape[0]
    mask = torch.rand(B, device=device) < prob
    if not mask.any():
        return batch

    t = torch.rand(B, device=device, dtype=batch["heatmap_norm"].dtype)
    t = t.clamp(1e-4, 1.0 - 1e-4)
    while t.dim() < batch["heatmap_norm"].dim():
        t = t.unsqueeze(-1)

    hm = batch["heatmap_norm"]
    hm_alt = batch["heatmap_norm_alt"]
    if hm.dim() == 3:
        hm = hm.unsqueeze(1)
    if hm_alt.dim() == 3:
        hm_alt = hm_alt.unsqueeze(1)

    pi = batch["PI_freq"].float()
    pi_alt = batch["PI_freq_alt"].float()
    hm_blend = (1.0 - t) * hm + t * hm_alt
    pi_blend = (1.0 - t.squeeze()) * pi + t.squeeze() * pi_alt

    out = dict(batch)
    hm_out = hm.clone()
    pi_out = pi.clone()
    hm_out[mask] = hm_blend[mask]
    pi_out[mask] = pi_blend[mask]
    out["heatmap_norm"] = hm_out
    out["PI_freq"] = pi_out
    return out
