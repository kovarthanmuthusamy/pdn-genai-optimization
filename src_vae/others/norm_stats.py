"""Centralized normalization / denormalization from ``normalization_stats.json``.

Supports:
  - ``log_zscore`` / ``log_zscore_unbounded`` (legacy mean/std on log1p)
  - ``robust_log1p_per_mhz`` / ``robust_log1p_per_mhz_unbounded`` (median/IQR per anchor)

All heatmap and impedance scale/denorm should go through this module.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src_vae.others.multifreq_anchors import load_anchors_mhz, nearest_anchor_mhz
from src_vae.others.pi_freq_utils import _LOG10_MIN, _LOG10_RANGE


def _mhz_key(mhz: float) -> str:
    return f"{float(mhz):.1f}"


def pi_norm_to_mhz(pi_norm: float | np.ndarray | torch.Tensor) -> np.ndarray | float:
    if isinstance(pi_norm, torch.Tensor):
        x = pi_norm.detach().float().cpu().numpy()
        log10_hz = x * _LOG10_RANGE + _LOG10_MIN
        return (10.0 ** log10_hz) / 1e6
    arr = np.asarray(pi_norm, dtype=np.float64)
    log10_hz = arr * _LOG10_RANGE + _LOG10_MIN
    return (10.0 ** log10_hz) / 1e6


@dataclass
class HeatmapBinStats:
    median: float
    iqr: float
    clip_min: float
    clip_max: float
    background_value: float
    # legacy z-score fields (filled for log_zscore mode)
    log_mean: float = 0.0
    log_std: float = 1.0
    z_max: float | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HeatmapBinStats:
        return cls(
            median=float(d.get("median", d.get("log_mean", 0.0))),
            iqr=float(d.get("iqr", d.get("log_std", 1.0))) or 1.0,
            clip_min=float(d["clip_min"]),
            clip_max=float(d["clip_max"]),
            background_value=float(d.get("background_value", d["clip_min"] - 1.5)),
            log_mean=float(d.get("log_mean", d.get("median", 0.0))),
            log_std=float(d.get("log_std", d.get("iqr", 1.0))) or 1.0,
            z_max=float(d["z_max"]) if d.get("z_max") is not None else None,
        )


@dataclass
class HeatmapNormStats:
    norm_mode: str
    background_value: float
    clip_min: float
    clip_max: float
    log_mean: float = 0.0
    log_std: float = 1.0
    by_mhz: dict[str, HeatmapBinStats] = field(default_factory=dict)
    anchors_mhz: tuple[float, ...] = ()
    global_max_ohm: float | None = None
    unbounded: bool = False
    z_max: float | None = None

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> HeatmapNormStats:
        hm = raw.get("Heatmap") or raw
        mode = str(hm.get("norm_mode", "log_zscore"))
        bg = float(raw.get("background_value", hm.get("background_value", -3.0)))
        by: dict[str, HeatmapBinStats] = {}
        if "by_mhz" in hm:
            for k, v in hm["by_mhz"].items():
                by[str(k)] = HeatmapBinStats.from_dict(v)
        unbounded = bool(hm.get("unbounded", False)) or mode.endswith("_unbounded")
        return cls(
            norm_mode=mode,
            background_value=bg,
            clip_min=float(hm.get("clip_min", -2.0)),
            clip_max=float(hm.get("clip_max", 4.0)),
            log_mean=float(hm.get("log_mean", 0.0)),
            log_std=float(hm.get("log_std", 1.0)) or 1.0,
            by_mhz=by,
            anchors_mhz=tuple(float(x) for x in hm.get("anchors_mhz", load_anchors_mhz())),
            global_max_ohm=(
                float(hm["global_max_ohm"]) if hm.get("global_max_ohm") is not None else None
            ),
            unbounded=unbounded,
            z_max=float(hm["z_max"]) if hm.get("z_max") is not None else None,
        )

    def is_unbounded(self) -> bool:
        return self.unbounded or self.norm_mode.endswith("_unbounded")

    def is_robust_per_mhz(self) -> bool:
        return (
            self.norm_mode in ("robust_log1p_per_mhz", "robust_log1p_per_mhz_unbounded")
            and bool(self.by_mhz)
        )

    @classmethod
    def load(cls, data_dir: str | Path, *, stats_path: str | Path | None = None) -> HeatmapNormStats:
        if stats_path is None:
            stats_path = Path(data_dir) / "normalization_stats.json"
        raw = json.loads(Path(stats_path).read_text(encoding="utf-8"))
        return cls.from_json(raw)

    def _use_clip(self, apply_clip: bool) -> bool:
        return apply_clip and not self.is_unbounded()

    def is_global_max(self) -> bool:
        return self.norm_mode == "global_max" or self.global_max_ohm is not None

    def resolve_mhz(
        self,
        mhz: float | None = None,
        *,
        pi_norm: float | np.ndarray | torch.Tensor | None = None,
    ) -> float:
        if mhz is not None:
            return float(nearest_anchor_mhz(float(mhz), self.anchors_mhz or load_anchors_mhz()))
        if pi_norm is not None:
            m = pi_norm_to_mhz(pi_norm)
            if isinstance(m, np.ndarray):
                return float(nearest_anchor_mhz(float(m.reshape(-1)[0]), self.anchors_mhz or load_anchors_mhz()))
            return float(nearest_anchor_mhz(float(m), self.anchors_mhz or load_anchors_mhz()))
        return float((self.anchors_mhz or load_anchors_mhz())[-1])

    def bin_stats(
        self,
        mhz: float | None = None,
        *,
        pi_norm=None,
        interp: bool = False,
    ) -> HeatmapBinStats:
        if self.is_robust_per_mhz():
            anchors = self.anchors_mhz or load_anchors_mhz()
            # True (un-snapped) MHz for interpolation decisions.
            if mhz is not None:
                m_true = float(mhz)
            elif pi_norm is not None:
                _m = pi_norm_to_mhz(pi_norm)
                m_true = float(_m.reshape(-1)[0]) if isinstance(_m, np.ndarray) else float(_m)
            else:
                m_true = float(anchors[-1])
            snapped = float(nearest_anchor_mhz(m_true, anchors))
            # Off-anchor + interp: linearly interpolate stats in log10(MHz) between
            # the two bracketing anchors. Anchor-exact frequencies (training path)
            # skip this and return their exact stats unchanged.
            if interp and abs(m_true - snapped) > 1e-6 and len(self.by_mhz) >= 2:
                return self._interp_bin_stats(m_true)
            key = _mhz_key(snapped)
            if key in self.by_mhz:
                return self.by_mhz[key]
            best = min(self.by_mhz.keys(), key=lambda k: abs(float(k) - m_true))
            return self.by_mhz[best]
        return HeatmapBinStats(
            median=self.log_mean,
            iqr=self.log_std,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
            background_value=self.background_value,
            log_mean=self.log_mean,
            log_std=self.log_std,
        )

    def _interp_bin_stats(self, mhz: float) -> HeatmapBinStats:
        """Log-frequency linear interpolation of per-MHz stats for off-anchor MHz.

        Fixes off-anchor magnitude: nearest-anchor snapping crushes the scale when
        the closest anchor is a low/degenerate band (e.g. 30 MHz -> 10 MHz ceiling
        ~0.44 Ohm). Interpolating median/IQR/clip in log10(MHz) between the two
        bracketing anchors restores the correct magnitude. Endpoints clamp.
        """
        anchor_keys = sorted(self.by_mhz.keys(), key=lambda k: float(k))
        vals = [float(k) for k in anchor_keys]
        if mhz <= vals[0]:
            return self.by_mhz[anchor_keys[0]]
        if mhz >= vals[-1]:
            return self.by_mhz[anchor_keys[-1]]
        lo = max(v for v in vals if v <= mhz)
        hi = min(v for v in vals if v >= mhz)
        if lo == hi:
            return self.by_mhz[_mhz_key(lo)]
        bl = self.by_mhz[_mhz_key(lo)]
        bh = self.by_mhz[_mhz_key(hi)]
        t = (math.log10(mhz) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))

        def lerp(a: float, b: float) -> float:
            return float(a + t * (b - a))

        if bl.z_max is not None and bh.z_max is not None:
            z_max = lerp(bl.z_max, bh.z_max)
        else:
            z_max = bl.z_max if bl.z_max is not None else bh.z_max
        return HeatmapBinStats(
            median=lerp(bl.median, bh.median),
            iqr=lerp(bl.iqr, bh.iqr) or 1.0,
            clip_min=lerp(bl.clip_min, bh.clip_min),
            clip_max=lerp(bl.clip_max, bh.clip_max),
            background_value=lerp(bl.background_value, bh.background_value),
            log_mean=lerp(bl.log_mean, bh.log_mean),
            log_std=lerp(bl.log_std, bh.log_std) or 1.0,
            z_max=z_max,
        )

    def clip_bounds(
        self,
        mhz: float | None = None,
        *,
        pi_norm=None,
    ) -> tuple[float, float]:
        b = self.bin_stats(mhz, pi_norm=pi_norm)
        return b.clip_min, b.clip_max

    def fg_threshold(self, margin: float = 0.5, mhz=None, *, pi_norm=None) -> float:
        b = self.bin_stats(mhz, pi_norm=pi_norm)
        return float(b.background_value) + margin

    def norm_to_physical_np(
        self,
        hm_norm: np.ndarray,
        *,
        mhz: float | None = None,
        pi_norm: float | None = None,
        apply_clip: bool = True,
    ) -> np.ndarray:
        if self.is_global_max():
            gmax = float(self.global_max_ohm or 1.0)
            z = hm_norm
            if apply_clip:
                lo, hi = self.clip_min, self.clip_max
                z = np.clip(z, lo, hi)
            return np.clip(z, 0.0, None) * gmax

        b = self.bin_stats(mhz, pi_norm=pi_norm)
        z = np.asarray(hm_norm, dtype=np.float64)
        if self._use_clip(apply_clip):
            z = np.clip(z, b.clip_min, b.clip_max)
        if self.is_robust_per_mhz():
            log1p = z * b.iqr + b.median
        else:
            log1p = z * b.log_std + b.log_mean
        return np.expm1(log1p).clip(min=0.0)

    def norm_to_physical(
        self,
        hm_norm: torch.Tensor,
        *,
        mhz: float | torch.Tensor | None = None,
        pi_norm: torch.Tensor | None = None,
        apply_clip: bool = True,
    ) -> torch.Tensor:
        if self.is_global_max():
            gmax = float(self.global_max_ohm or 1.0)
            z = hm_norm
            if apply_clip:
                z = z.clamp(self.clip_min, self.clip_max)
            return z.clamp(min=0.0) * gmax

        if hm_norm.dim() == 3:
            hm_norm = hm_norm.unsqueeze(1)

        if mhz is not None and isinstance(mhz, torch.Tensor) and mhz.numel() > 1:
            return self.norm_to_physical_batch(hm_norm, mhz=mhz, apply_clip=apply_clip)

        mhz_f = None
        if mhz is not None:
            mhz_f = float(mhz.item() if isinstance(mhz, torch.Tensor) else mhz)
        elif pi_norm is not None:
            mhz_f = self.resolve_mhz(pi_norm=pi_norm)

        b = self.bin_stats(mhz_f, pi_norm=pi_norm, interp=True)
        z = hm_norm
        if self._use_clip(apply_clip):
            z = z.clamp(b.clip_min, b.clip_max)
        if self.is_robust_per_mhz():
            log1p = z * b.iqr + b.median
        else:
            log1p = z * b.log_std + b.log_mean
        return (torch.exp(log1p) - 1.0).clamp(min=0.0)

    def norm_to_physical_batch(
        self,
        hm_norm: torch.Tensor,
        *,
        mhz: torch.Tensor,
        apply_clip: bool = True,
    ) -> torch.Tensor:
        """Per-sample MHz denorm (B,1,H,W)."""
        bsz = hm_norm.shape[0]
        out = []
        for i in range(bsz):
            m = float(mhz[i].item()) if mhz.ndim else float(mhz.item())
            out.append(
                self.norm_to_physical(hm_norm[i : i + 1], mhz=m, apply_clip=apply_clip)
            )
        return torch.cat(out, dim=0)

    def physical_ceiling_ohm(self, mhz: float | None = None) -> float | None:
        if self.is_global_max():
            return float(self.global_max_ohm or 0.0) * float(self.clip_max)
        b = self.bin_stats(mhz)
        if self.is_unbounded():
            z_hi = float(b.z_max if b.z_max is not None else (self.z_max if self.z_max is not None else b.clip_max))
        else:
            z_hi = b.clip_max
        if self.is_robust_per_mhz():
            return float(np.expm1(z_hi * b.iqr + b.median))
        return float(np.expm1(z_hi * b.log_std + b.log_mean))

    def physical_ceiling_ohm_soft(self, mhz: float | None = None) -> float | None:
        """p99.5 reference ceiling (metadata) — use for QC flags when unbounded."""
        b = self.bin_stats(mhz)
        z_hi = b.clip_max
        if self.is_robust_per_mhz():
            return float(np.expm1(z_hi * b.iqr + b.median))
        return float(np.expm1(z_hi * b.log_std + b.log_mean))

    def describe(self) -> str:
        ub = " unbounded" if self.is_unbounded() else ""
        if self.is_robust_per_mhz():
            keys = ", ".join(sorted(self.by_mhz.keys())[:6])
            return f"{self.norm_mode}{ub}  bins=[{keys}…]  n={len(self.by_mhz)}"
        if self.is_global_max():
            return f"global_max  gmax={self.global_max_ohm:.4f}Ω"
        return (
            f"{self.norm_mode}{ub}  mean={self.log_mean:.4f} std={self.log_std:.4f} "
            f"clip=[{self.clip_min:.3f},{self.clip_max:.3f}]"
        )


@dataclass
class ImpedanceNormStats:
    log_mean: float
    log_std: float

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> ImpedanceNormStats:
        imp = raw.get("Impedance") or raw
        return cls(
            log_mean=float(imp["log_mean"]),
            log_std=float(imp.get("log_std", 1.0)) or 1.0,
        )

    @classmethod
    def load(cls, data_dir: str | Path, *, stats_path: str | Path | None = None) -> ImpedanceNormStats:
        if stats_path is None:
            stats_path = Path(data_dir) / "normalization_stats.json"
        raw = json.loads(Path(stats_path).read_text(encoding="utf-8"))
        return cls.from_json(raw)

    def norm_to_log(self, imp_norm: torch.Tensor) -> torch.Tensor:
        z = imp_norm[:, 0] if imp_norm.dim() == 3 else imp_norm
        return z * self.log_std + self.log_mean

    def norm_to_ohm(self, imp_norm: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.norm_to_log(imp_norm))


@dataclass
class NormStatsBundle:
    heatmap: HeatmapNormStats
    impedance: ImpedanceNormStats
    raw: dict[str, Any]

    @classmethod
    def load(cls, data_dir: str | Path, *, stats_path: str | Path | None = None) -> NormStatsBundle:
        if stats_path is None:
            stats_path = Path(data_dir) / "normalization_stats.json"
        raw = json.loads(Path(stats_path).read_text(encoding="utf-8"))
        return cls(
            heatmap=HeatmapNormStats.from_json(raw),
            impedance=ImpedanceNormStats.from_json(raw),
            raw=raw,
        )


def load_norm_stats(data_dir: str | Path, *, stats_path: str | Path | None = None) -> NormStatsBundle:
    return NormStatsBundle.load(data_dir, stats_path=stats_path)


def load_heatmap_stats(data_dir: str | Path, *, stats_path: str | Path | None = None) -> HeatmapNormStats:
    return HeatmapNormStats.load(data_dir, stats_path=stats_path)


def load_heatmap_clip_bounds(data_dir: str | Path, *, stats_path: str | Path | None = None) -> tuple[float, float] | None:
    """Global clip bounds for training. Returns None when stats are unbounded."""
    try:
        hm = HeatmapNormStats.load(data_dir, stats_path=stats_path)
        if hm.is_unbounded():
            return None
        if hm.is_robust_per_mhz():
            mins = [b.clip_min for b in hm.by_mhz.values()]
            maxs = [b.clip_max for b in hm.by_mhz.values()]
            return (min(mins), max(maxs))
        return hm.clip_min, hm.clip_max
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None
