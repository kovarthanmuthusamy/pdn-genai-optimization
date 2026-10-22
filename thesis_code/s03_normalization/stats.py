"""Read the normalization statistics and map normalized values back to ohm.

What: normalization_stats.json holds, for maps, a median and an IQR of log(1 + ohm) per frequency bin
    ("by_mhz"; mode robust_log1p_per_mhz_unbounded, values not clipped), and for spectra one mean and
    standard deviation of log(ohm). At a frequency between two bin centres the map statistics are
    interpolated linearly in log10(MHz) (``bin_stats(mhz, interp=True)``, used when normalizing and
    in the model's map read-out); without ``interp`` the nearest bin centre is used.
Usage (import only):
    load_norm_stats(dir_or_file_or_None, stats_path=None) -> NormStatsBundle (.heatmap, .impedance, .raw);
        None reads common/inputs/normalization_stats.json
    HeatmapNormStats.bin_stats / norm_to_physical (torch) / norm_to_physical_np / norm_to_physical_batch
    ImpedanceNormStats.norm_to_log / norm_to_ohm
    pi_norm_to_mhz(model frequency input) -> MHz
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from thesis_code.common.frequency import _LOG10_MIN, _LOG10_RANGE, INPUTS, nearest_anchor_mhz

NORM_MODE = "robust_log1p_per_mhz_unbounded"


def _mhz_key(mhz: float) -> str:
    return f"{float(mhz):.1f}"


def pi_norm_to_mhz(pi_norm: float | np.ndarray | torch.Tensor) -> np.ndarray | float:
    if isinstance(pi_norm, torch.Tensor):
        x = pi_norm.detach().float().cpu().numpy()
        return (10.0 ** (x * _LOG10_RANGE + _LOG10_MIN)) / 1e6
    arr = np.asarray(pi_norm, dtype=np.float64)
    return (10.0 ** (arr * _LOG10_RANGE + _LOG10_MIN)) / 1e6


@dataclass
class HeatmapBinStats:
    median: float
    iqr: float
    clip_min: float
    clip_max: float
    background_value: float
    z_max: float | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HeatmapBinStats:
        return cls(median=float(d["median"]), iqr=float(d["iqr"]) or 1.0, clip_min=float(d["clip_min"]),
                   clip_max=float(d["clip_max"]), background_value=float(d["background_value"]),
                   z_max=float(d["z_max"]) if d.get("z_max") is not None else None)


@dataclass
class HeatmapNormStats:
    background_value: float
    clip_min: float
    clip_max: float
    by_mhz: dict[str, HeatmapBinStats] = field(default_factory=dict)
    anchors_mhz: tuple[float, ...] = ()
    z_max: float | None = None

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> HeatmapNormStats:
        hm = raw["Heatmap"]
        if hm.get("norm_mode") != NORM_MODE or not hm.get("by_mhz"):
            raise ValueError(f"expected per-frequency statistics in mode {NORM_MODE}, got {hm.get('norm_mode')}")
        return cls(background_value=float(raw.get("background_value", hm["background_value"])),
                   clip_min=float(hm["clip_min"]), clip_max=float(hm["clip_max"]),
                   by_mhz={str(k): HeatmapBinStats.from_dict(v) for k, v in hm["by_mhz"].items()},
                   anchors_mhz=tuple(float(x) for x in hm["anchors_mhz"]),
                   z_max=float(hm["z_max"]) if hm.get("z_max") is not None else None)

    def bin_stats(self, mhz: float | None = None, *, pi_norm=None, interp: bool = False) -> HeatmapBinStats:
        """Statistics at ``mhz`` (or at the frequency of ``pi_norm``): the nearest bin centre, or with
        ``interp`` an interpolation between the two bracketing centres."""
        if mhz is not None:
            m_true = float(mhz)
        elif pi_norm is not None:
            _m = pi_norm_to_mhz(pi_norm)
            m_true = float(_m.reshape(-1)[0]) if isinstance(_m, np.ndarray) else float(_m)
        else:
            m_true = float(self.anchors_mhz[-1])
        snapped = float(nearest_anchor_mhz(m_true, self.anchors_mhz))
        if interp and abs(m_true - snapped) > 1e-6 and len(self.by_mhz) >= 2:
            return self._interp_bin_stats(m_true)
        key = _mhz_key(snapped)
        if key in self.by_mhz:
            return self.by_mhz[key]
        return self.by_mhz[min(self.by_mhz.keys(), key=lambda k: abs(float(k) - m_true))]

    def _interp_bin_stats(self, mhz: float) -> HeatmapBinStats:
        """Linear interpolation in log10(MHz) between the bracketing bin centres; ends clamp."""
        keys = sorted(self.by_mhz.keys(), key=float)
        vals = [float(k) for k in keys]
        if mhz <= vals[0]:
            return self.by_mhz[keys[0]]
        if mhz >= vals[-1]:
            return self.by_mhz[keys[-1]]
        lo = max(v for v in vals if v <= mhz)
        hi = min(v for v in vals if v >= mhz)
        if lo == hi:
            return self.by_mhz[_mhz_key(lo)]
        bl, bh = self.by_mhz[_mhz_key(lo)], self.by_mhz[_mhz_key(hi)]
        t = (math.log10(mhz) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))

        def lerp(a: float, b: float) -> float:
            return float(a + t * (b - a))

        if bl.z_max is not None and bh.z_max is not None:
            z_max = lerp(bl.z_max, bh.z_max)
        else:
            z_max = bl.z_max if bl.z_max is not None else bh.z_max
        return HeatmapBinStats(median=lerp(bl.median, bh.median), iqr=lerp(bl.iqr, bh.iqr) or 1.0,
                               clip_min=lerp(bl.clip_min, bh.clip_min), clip_max=lerp(bl.clip_max, bh.clip_max),
                               background_value=lerp(bl.background_value, bh.background_value), z_max=z_max)

    def norm_to_physical_np(self, hm_norm: np.ndarray, *, mhz: float | None = None,
                            pi_norm: float | None = None) -> np.ndarray:
        """Normalized map -> ohm with the nearest bin's statistics."""
        b = self.bin_stats(mhz, pi_norm=pi_norm)
        y = np.asarray(hm_norm, dtype=np.float64) * b.iqr + b.median
        return np.expm1(y).clip(min=0.0)

    def norm_to_physical(self, hm_norm: torch.Tensor, *, mhz: float | torch.Tensor | None = None,
                         pi_norm: torch.Tensor | None = None) -> torch.Tensor:
        """Normalized map(s) -> ohm with interpolated statistics; a tensor ``mhz`` gives one per sample."""
        if hm_norm.dim() == 3:
            hm_norm = hm_norm.unsqueeze(1)
        if mhz is not None and isinstance(mhz, torch.Tensor) and mhz.numel() > 1:
            return self.norm_to_physical_batch(hm_norm, mhz=mhz)
        mhz_f = None
        if mhz is not None:
            mhz_f = float(mhz.item() if isinstance(mhz, torch.Tensor) else mhz)
        elif pi_norm is not None:
            m = pi_norm_to_mhz(pi_norm)
            m = float(m.reshape(-1)[0]) if isinstance(m, np.ndarray) else float(m)
            mhz_f = float(nearest_anchor_mhz(m, self.anchors_mhz))
        b = self.bin_stats(mhz_f, pi_norm=pi_norm, interp=True)
        return (torch.exp(hm_norm * b.iqr + b.median) - 1.0).clamp(min=0.0)

    def norm_to_physical_batch(self, hm_norm: torch.Tensor, *, mhz: torch.Tensor) -> torch.Tensor:
        """Per-sample frequency read-out (B, 1, H, W)."""
        out = [self.norm_to_physical(hm_norm[i:i + 1], mhz=float(mhz[i].item()) if mhz.ndim else float(mhz.item()))
               for i in range(hm_norm.shape[0])]
        return torch.cat(out, dim=0)


@dataclass
class ImpedanceNormStats:
    log_mean: float
    log_std: float

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> ImpedanceNormStats:
        imp = raw["Impedance"]
        return cls(log_mean=float(imp["log_mean"]), log_std=float(imp.get("log_std", 1.0)) or 1.0)

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


def load_norm_stats(data_dir: str | Path | None = None, *, stats_path: str | Path | None = None) -> NormStatsBundle:
    """Statistics of a dataset folder, a statistics file, or ``stats_path``; with none of them the
    shipped copy in common/inputs/."""
    if stats_path is None:
        base = Path(data_dir) if data_dir is not None else INPUTS
        stats_path = base if base.is_file() else base / "normalization_stats.json"
    raw = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    return NormStatsBundle(heatmap=HeatmapNormStats.from_json(raw), impedance=ImpedanceNormStats.from_json(raw), raw=raw)
