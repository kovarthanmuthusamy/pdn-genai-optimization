"""Monkey-patch exp038 trainer for global-max heatmap normalization (exp043).

Run:
    Called automatically from ``train_vae_simple._on_stats_loaded`` — not run directly."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from experiments.exp043.codes.gmax_heatmap_loss import (
    cross_freq_heatmap_loss_gmax,
    heatmap_loss_gmax,
)
from src_vae.others.heatmap_gmax_norm import (
    LOG1P_TRAIN_SPACE,
    disk_to_train_space,
    heatmap_fg_threshold,
    heatmap_phys_amplitude_loss_gmax,
    is_global_max_stats,
    is_log1p_train_space,
    linear_clip_bounds_to_train,
    linear_threshold_to_train,
    load_gmax_from_stats,
)


def load_heatmap_stats(data_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(data_dir) / "normalization_stats.json").read_text(encoding="utf-8"))


def apply_gmax_config(c, raw: dict[str, Any] | None = None) -> bool:
    """If dataset uses global-max heatmaps, set clip/bg/phys fields on ``c``. Returns True if applied."""
    data_dir = Path(c.data_dir)
    if raw is None:
        p = data_dir / "normalization_stats.json"
        if not p.is_file():
            return False
        raw = json.loads(p.read_text(encoding="utf-8"))
    hm = raw.get("Heatmap") or {}
    if not is_global_max_stats(raw):
        return False

    gmax, _bg = load_gmax_from_stats(raw)
    fg_thr_linear = float(hm.get("fg_norm_threshold", 0.01))
    cfg_thr = getattr(c, "heatmap_fg_threshold", None)
    if cfg_thr is not None:
        fg_thr_linear = float(cfg_thr)

    clip_lo = float(hm["clip_min"])
    clip_hi = float(hm["clip_max"])
    c.use_global_max_heatmap = True
    c.global_max_ohm = gmax
    c.heatmap_fg_threshold_linear = fg_thr_linear
    c.heatmap_disk_clip_min = clip_lo
    c.heatmap_disk_clip_max = clip_hi

    train_space = getattr(c, "heatmap_train_space", "linear")
    if train_space == LOG1P_TRAIN_SPACE:
        c.heatmap_fg_threshold = linear_threshold_to_train(fg_thr_linear, gmax)
        c.heatmap_z_clip_min, c.heatmap_z_clip_max = linear_clip_bounds_to_train(
            clip_lo, clip_hi, gmax,
        )
    else:
        c.heatmap_fg_threshold = fg_thr_linear
        c.heatmap_z_clip_min = clip_lo
        c.heatmap_z_clip_max = clip_hi

    c.background_value = float(c.heatmap_fg_threshold) - 0.5
    c.physics_fg_clip_min = float(c.heatmap_z_clip_min)
    return True


def _prepare_batch_gmax(batch: dict, c, orig_prepare):
    """Synthetic blend + disk→train remap + FG mask in train space."""
    from experiments.exp043.codes.synthetic_freq_blend import maybe_apply_synthetic_blend
    from src_vae.others.pi_freq_utils import pi_freq_mhz_to_norm

    batch = maybe_apply_synthetic_blend(batch, c)
    nb = c.is_cuda()
    dev = c.device
    hm_lin = batch["heatmap_norm"].to(dev, non_blocking=nb)
    occ = batch["occupancy"].to(dev, non_blocking=nb)
    imp = batch["impedance"].to(dev, non_blocking=nb)
    K = batch["K"].to(dev, non_blocking=nb)
    if "PI_freq" in batch:
        pi = batch["PI_freq"].to(dev, non_blocking=nb)
    else:
        pi = torch.full((hm_lin.shape[0],), pi_freq_mhz_to_norm(200.0), device=dev)

    if hm_lin.dim() == 3:
        hm_lin = hm_lin.unsqueeze(1)
    if imp.dim() == 1:
        imp = imp.unsqueeze(0)
    if imp.dim() == 2 and imp.shape[-1] == 231:
        imp = imp.unsqueeze(1)
    elif imp.dim() == 3:
        imp = imp[:, :1]

    lo_d = float(getattr(c, "heatmap_disk_clip_min", 0.0))
    hi_d = float(getattr(c, "heatmap_disk_clip_max", 1.02))
    hm_lin = hm_lin.clamp(lo_d, hi_d)
    hm = disk_to_train_space(hm_lin, c)
    lo_t, hi_t = c.heatmap_z_clip_min, c.heatmap_z_clip_max
    if lo_t is not None and hi_t is not None:
        hm = hm.clamp(lo_t, hi_t)
    thr = heatmap_fg_threshold(c)
    hm = hm.masked_fill(hm < thr, 0.0)
    hm_enc = hm.clone()
    return hm, hm_enc, occ, imp, K, pi


def transform_disk_heatmap(hm: torch.Tensor, c) -> torch.Tensor:
    """Linear on-disk heatmap batch → train space (for cross-freq alt / eval)."""
    if hm.dim() == 3:
        hm = hm.unsqueeze(1)
    lo_d = float(getattr(c, "heatmap_disk_clip_min", 0.0))
    hi_d = float(getattr(c, "heatmap_disk_clip_max", 1.02))
    hm = hm.clamp(lo_d, hi_d)
    hm = disk_to_train_space(hm, c)
    lo_t, hi_t = c.heatmap_z_clip_min, c.heatmap_z_clip_max
    if lo_t is not None and hi_t is not None:
        hm = hm.clamp(lo_t, hi_t)
    thr = heatmap_fg_threshold(c)
    return hm.masked_fill(hm < thr, 0.0)


def patch_trainer(_tr, c) -> None:
    """Monkey-patch exp038 trainer losses when ``c`` uses global-max heatmaps."""
    if not getattr(c, "use_global_max_heatmap", False):
        return

    gmax_ohm = getattr(c, "global_max_ohm", None)
    if gmax_ohm is None:
        raise ValueError(
            "global_max_ohm is unset — must be loaded from normalization_stats.json "
            "via apply_gmax_config() before patch_trainer().",
        )
    gmax = float(gmax_ohm)
    orig_prepare = _tr._prepare_batch

    def heatmap_phys_amplitude_loss(recon, target, hm_log_mean, hm_log_std, cfg):
        return heatmap_phys_amplitude_loss_gmax(recon, target, gmax, cfg)

    def heatmap_loss(recon, target, cfg, ps, *, dynrange_weight=None, lite=False):
        return heatmap_loss_gmax(
            recon, target, cfg, ps, dynrange_weight=dynrange_weight, lite=lite,
        )

    def _cross_freq_heatmap_loss(
        model, z, K, pi_alt, hm_alt, cfg, ps, *,
        dynrange_weight=None, occupancy=None, heatmap_skips=None,
    ):
        return cross_freq_heatmap_loss_gmax(
            model, z, K, pi_alt, hm_alt, cfg, ps,
            dynrange_weight=dynrange_weight, occupancy=occupancy,
            heatmap_skips=heatmap_skips,
        )

    def _prepare_batch(batch, cfg):
        return _prepare_batch_gmax(batch, cfg, orig_prepare)

    _tr.heatmap_phys_amplitude_loss = heatmap_phys_amplitude_loss
    _tr.heatmap_loss = heatmap_loss
    _tr._cross_freq_heatmap_loss = _cross_freq_heatmap_loss
    _tr._prepare_batch = _prepare_batch
    _tr.transform_disk_heatmap = lambda hm, cfg: transform_disk_heatmap(hm, cfg)

    full_cf = bool(getattr(c, "cross_freq_full_loss", True))
    ts = getattr(c, "heatmap_train_space", "linear")
    print(
        f"  exp043 global-max patches: gmax={gmax:.4f} Ω  train_space={ts}  "
        f"fg_thr={c.heatmap_fg_threshold:.6f} (linear={getattr(c, 'heatmap_fg_threshold_linear', c.heatmap_fg_threshold):.6f})  "
        f"clip=[{c.heatmap_z_clip_min:.4f}, {c.heatmap_z_clip_max:.4f}]  "
        f"huberδ={getattr(c, 'heatmap_huber_delta', 0.2)}  "
        f"pattern_w={getattr(c, 'heatmap_pattern_weight', 0.0)}  "
        f"spread_w={getattr(c, 'heatmap_spread_weight', 0.0)}  "
        f"hotspot_w={getattr(c, 'heatmap_hotspot_weight', 0.0)}  "
        f"centroid_w={getattr(c, 'heatmap_peak_centroid_weight', 0.0)}  "
        f"quantile_w={getattr(c, 'heatmap_quantile_weight', 0.0)}  "
        f"cross_freq_full={full_cf}",
    )
