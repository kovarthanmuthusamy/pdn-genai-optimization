"""Patch shared trainer for global-max heatmap normalization (exp043)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src_vae.others.heatmap_gmax_norm import (
    heatmap_phys_amplitude_loss_gmax,
    is_global_max_stats,
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

    gmax, bg = load_gmax_from_stats(raw)
    cfg_thr = getattr(c, "heatmap_fg_threshold", None)
    fg_thr = float(cfg_thr if cfg_thr is not None else hm.get("fg_norm_threshold", 0.01))
    c.use_global_max_heatmap = True
    c.global_max_ohm = gmax
    c.heatmap_fg_threshold = fg_thr
    c.background_value = fg_thr - 0.5
    c.heatmap_z_clip_min = float(hm["clip_min"])
    c.heatmap_z_clip_max = float(hm["clip_max"])
    c.physics_fg_clip_min = float(hm["clip_min"])
    if raw.get("background_value") is not None:
        raw_bg = float(raw["background_value"])
        if abs(raw_bg) < 1e-6:
            pass
    return True


def patch_trainer(_tr, c) -> None:
    """Monkey-patch exp038 trainer losses when ``c`` uses global-max heatmaps."""
    if not getattr(c, "use_global_max_heatmap", False):
        return

    gmax = float(c.global_max_ohm)

    def heatmap_phys_amplitude_loss(recon, target, hm_log_mean, hm_log_std, cfg):
        return heatmap_phys_amplitude_loss_gmax(recon, target, gmax, cfg)

    _tr.heatmap_phys_amplitude_loss = heatmap_phys_amplitude_loss
    print(
        f"  exp043 global-max heatmap: gmax={gmax:.4f} Ω  "
        f"fg_thr={c.heatmap_fg_threshold:.4f}  clip=[{c.heatmap_z_clip_min}, {c.heatmap_z_clip_max}]",
    )
