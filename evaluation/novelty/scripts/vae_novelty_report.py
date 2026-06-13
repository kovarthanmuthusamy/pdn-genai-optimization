#!/usr/bin/env python3
"""Nearest-neighbor novelty / memorization check for this repo's multi-modal VAE.

Goal
- For each generated sample, find the closest training sample (kNN with k=1)
  and report distances for:
    - pooled heatmap z-score (MSE)
    - impedance log-zscore (MSE)
    - occupancy vector (Hamming distance)
  plus a simple combined score.

Interpretation (rule of thumb)
- If occupancy matches *exactly* (Hamming=0) and the continuous distances are
  extremely small, you're likely memorizing.
- Compare generated→train NN distances to train→train NN baseline distances.
  If generated distances are consistently much smaller than baseline, that is
  suspicious.

This is not a proof of "novel" physics; it is a practical "are we copying the
training set?" sanity check.

Additional novelty method (discrete)
- Occupancy-pattern novelty: for a fixed K, treat the 52-bit occupancy vector
    as a discrete pattern and measure how often generated patterns were never
    seen in the training set for that K.

Example
  python scripts/vae_novelty_report.py \
    --gen-dir scrap/generated_samples/K5 \
    --dataset-root datasets/data_norm \
    --K 5 \
    --hm-pool 16
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, TypedDict

import numpy as np


class Quantiles(TypedDict):
    min: float | None
    p10: float | None
    median: float | None
    p90: float | None
    max: float | None


class OccupancyPatternStats(TypedDict):
    train_unique: int
    gen_unique: int
    gen_new_unique: int
    gen_new_rate: float
    gen_mode_frac: float
    jsd: float
    gen_in_train: list[bool]
    train_entropy: float
    gen_entropy: float


@dataclass(frozen=True)
class DatasetArrays:
    names: list[str]
    k_values: np.ndarray  # (N,)
    heatmap_feat: np.ndarray  # (N, Dh)
    imp_z: np.ndarray  # (N, Di)
    occ_bin: np.ndarray  # (N, 52) uint8


@dataclass(frozen=True)
class ScoreSummary:
    n_train: int
    n_gen: int
    k_filter: int | None
    baseline_combined: Quantiles
    gen_to_train_combined: Quantiles
    gen_to_train_heatmap_mse: Quantiles
    gen_to_train_impedance_mse: Quantiles
    gen_to_train_occupancy_hamming: Quantiles
    occupancy_exact_count: int
    train_occ_unique: int | None
    gen_occ_unique: int | None
    gen_occ_new_unique: int | None
    gen_occ_new_rate: float | None
    gen_occ_mode_frac: float | None
    occ_jsd: float | None

    # Additional novelty / coverage methods
    # - Percentile-based memorization score: gen→train NN distance percentile
    #   against the train→train NN baseline distribution (lower = closer than
    #   typical train→train spacing).
    gen_to_train_combined_percentile: Quantiles | None
    mem_suspect_rate_p01: float | None
    mem_suspect_rate_p05: float | None
    mem_suspect_rate_p10: float | None

    # - Train coverage: for each training sample, distance to nearest generated
    #   sample in the same combined metric (lower = better coverage).
    train_to_gen_combined: Quantiles | None
    train_to_gen_coverage_at_base_p90: float | None


def _infer_k_from_path(path: Path) -> Optional[int]:
    m = re.search(r"(?:^|/)K(\d+)(?:/|$)", str(path).replace("\\", "/"))
    return int(m.group(1)) if m else None


def _iter_sample_dirs(gen_dir: Path) -> list[Path]:
    dirs = [p for p in gen_dir.glob("data_sample_*") if p.is_dir()]

    def _idx(p: Path) -> int:
        try:
            return int(p.name.split("_")[-1])
        except Exception:
            return 10**9

    return sorted(dirs, key=_idx)


def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _k_cache_path(dataset_root: Path) -> Path:
    return dataset_root / "k_index_cache.npz"


def _load_or_build_k_cache(dataset_root: Path, *, force_rebuild: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (stems, k_values) for the dataset, cached to disk.

    Cache file: <dataset_root>/k_index_cache.npz
      - stems: array[str] of length N
      - k:     array[int16] of length N

    This avoids rescanning all occupancy files for every K.
    """
    cache_path = _k_cache_path(dataset_root)
    if cache_path.exists() and not force_rebuild:
        data = np.load(cache_path, allow_pickle=False)
        return data["stems"], data["k"]

    occ_dir = dataset_root / "Occ_map"
    if not occ_dir.exists():
        raise SystemExit(f"Missing occupancy directory for K cache: {occ_dir}")

    occ_files = sorted(occ_dir.glob("*.npy"))
    if not occ_files:
        raise SystemExit(f"No occupancy .npy files found in {occ_dir}")

    stems: list[str] = []
    ks: list[int] = []
    for i, occ_path in enumerate(occ_files):
        occ = np.load(occ_path, mmap_mode="r").reshape(-1)
        k = int((occ > 0.5).sum())
        stems.append(occ_path.stem)
        ks.append(k)
        if (i + 1) % 2000 == 0:
            print(f"Scanned {i+1}/{len(occ_files)} occupancy files...")

    stems_arr = np.asarray(stems)
    k_arr = np.asarray(ks, dtype=np.int16)
    np.savez_compressed(cache_path, stems=stems_arr, k=k_arr)
    print(f"Wrote K cache: {cache_path}")
    return stems_arr, k_arr


def _pool_heatmap_zscore(hm: np.ndarray, pool: int) -> np.ndarray:
    """Mean-pool a (64,64) or (1,64,64) heatmap into (pool,pool), then flatten."""
    if hm.ndim == 3:
        hm = hm[0]
    if hm.ndim != 2:
        raise ValueError(f"Expected heatmap to be 2D after squeeze, got {hm.shape}")
    h, w = hm.shape
    if h != 64 or w != 64:
        raise ValueError(f"Expected heatmap shape (64,64), got {hm.shape}")
    if 64 % pool != 0:
        raise ValueError(f"hm-pool must divide 64; got {pool}")
    b = 64 // pool
    pooled = hm.reshape(pool, b, pool, b).mean(axis=(1, 3))
    return pooled.astype(np.float32, copy=False).reshape(-1)


def _load_occ_bin(path: Path) -> np.ndarray:
    occ = np.load(path)
    occ = occ.reshape(-1)
    if occ.size != 52:
        raise ValueError(f"Expected occupancy length 52, got {occ.shape} from {path}")
    return (occ > 0.5).astype(np.uint8)


def _load_impedance_channel0_z(dataset_imp_path: Path) -> np.ndarray:
    """Load dataset impedance and return channel 0 z-score vector (231,)."""
    imp = np.load(dataset_imp_path)
    if imp.ndim == 2:
        # expected (C,231)
        imp0 = imp[0]
    else:
        imp0 = imp.reshape(-1)
    if imp0.size != 231:
        raise ValueError(f"Expected impedance length 231, got {imp0.shape} from {dataset_imp_path}")
    return imp0.astype(np.float32, copy=False)


def _occ_keys(occ_bin: np.ndarray) -> list[bytes]:
    occ_bin = np.asarray(occ_bin)
    if occ_bin.ndim != 2 or occ_bin.shape[1] != 52:
        raise ValueError(f"Expected occupancy array of shape (N,52), got {occ_bin.shape}")
    occ_bin = occ_bin.astype(np.uint8, copy=False)
    packed = np.packbits(occ_bin, axis=1)
    return [packed[i].tobytes() for i in range(packed.shape[0])]


def _entropy_bits(counter: Counter[bytes], total: int) -> float:
    if total <= 0 or not counter:
        return 0.0
    p = np.asarray([c / float(total) for c in counter.values()], dtype=np.float64)
    return float(-(p * np.log2(p)).sum())


def _jsd_bits(p_counter: Counter[bytes], p_total: int, q_counter: Counter[bytes], q_total: int) -> float:
    if p_total <= 0 or q_total <= 0:
        return 0.0
    keys = list(set(p_counter.keys()) | set(q_counter.keys()))
    p = np.asarray([p_counter.get(k, 0) / float(p_total) for k in keys], dtype=np.float64)
    q = np.asarray([q_counter.get(k, 0) / float(q_total) for k in keys], dtype=np.float64)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())

    return 0.5 * (_kl(p, m) + _kl(q, m))


def _occupancy_pattern_stats(train_occ: np.ndarray, gen_occ: np.ndarray) -> OccupancyPatternStats:
    train_keys = _occ_keys(train_occ)
    gen_keys = _occ_keys(gen_occ)
    train_c = Counter(train_keys)
    gen_c = Counter(gen_keys)
    train_set = set(train_c.keys())

    new_keys = [k for k in gen_keys if k not in train_set]
    new_rate = float(len(new_keys)) / float(len(gen_keys)) if gen_keys else 0.0

    gen_mode_frac = 0.0
    if gen_c and gen_keys:
        gen_mode_frac = float(max(gen_c.values())) / float(len(gen_keys))

    return {
        "train_unique": int(len(train_c)),
        "gen_unique": int(len(gen_c)),
        "gen_new_unique": int(len(set(gen_c.keys()) - train_set)),
        "gen_new_rate": float(new_rate),
        "gen_mode_frac": float(gen_mode_frac),
        "jsd": float(_jsd_bits(train_c, len(train_keys), gen_c, len(gen_keys))),
        "gen_in_train": [k in train_set for k in gen_keys],
        "train_entropy": float(_entropy_bits(train_c, len(train_keys))),
        "gen_entropy": float(_entropy_bits(gen_c, len(gen_keys))),
    }


def _imp_log_to_z(imp_log: np.ndarray, *, log_mean: float, log_std: float) -> np.ndarray:
    imp_log = np.asarray(imp_log).reshape(-1)
    if imp_log.size != 231:
        raise ValueError(f"Expected generated impedance length 231, got {imp_log.shape}")
    return ((imp_log.astype(np.float32, copy=False) - float(log_mean)) / float(log_std)).astype(np.float32)


def _mse_matrix(gen: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Return MSE matrix (M,N) for L2 in feature space, using dot-product identity."""
    gen = gen.astype(np.float32, copy=False)
    train = train.astype(np.float32, copy=False)

    gen_norm2 = np.sum(gen * gen, axis=1, keepdims=True)  # (M,1)
    train_norm2 = np.sum(train * train, axis=1, keepdims=True).T  # (1,N)
    dots = gen @ train.T  # (M,N)
    dist2 = gen_norm2 + train_norm2 - 2.0 * dots
    # numerical safety
    dist2 = np.maximum(dist2, 0.0)
    return dist2 / float(train.shape[1])


def _hamming_matrix(gen_occ: np.ndarray, train_occ: np.ndarray) -> np.ndarray:
    """Compute (M,N) Hamming distances for 52-bit occupancy. Uses a loop over M."""
    gen_occ = gen_occ.astype(np.uint8, copy=False)
    train_occ = train_occ.astype(np.uint8, copy=False)
    m = gen_occ.shape[0]
    n = train_occ.shape[0]
    out = np.empty((m, n), dtype=np.uint8)
    for i in range(m):
        out[i] = np.count_nonzero(train_occ != gen_occ[i], axis=1).astype(np.uint8)
    return out


def load_dataset(
    *,
    dataset_root: Path,
    hm_pool: int,
    max_train: Optional[int],
    k_filter: Optional[int],
    seed: int,
) -> DatasetArrays:
    heatmap_dir = dataset_root / "heatmap"
    imp_dir = dataset_root / "Imp"
    occ_dir = dataset_root / "Occ_map"
    for d in (heatmap_dir, imp_dir, occ_dir):
        if not d.exists():
            raise SystemExit(f"Missing dataset directory: {d}")

    heatmap_files = sorted(heatmap_dir.glob("*.npy"))
    if not heatmap_files:
        raise SystemExit(f"No heatmap .npy files found in {heatmap_dir}")

    stems = [p.stem for p in heatmap_files]

    # Filter by K using a cached occupancy scan (fast after first run).
    if k_filter is not None:
        cache_stems, cache_k = _load_or_build_k_cache(dataset_root)
        k_mask = cache_k == int(k_filter)
        selected_stems = set(cache_stems[k_mask].tolist())
        selected: list[tuple[str, int]] = [(s, int(k_filter)) for s in stems if s in selected_stems]
    else:
        selected = [(stem, -1) for stem in stems]

    if not selected:
        raise SystemExit(f"No dataset samples matched K={k_filter}")

    if max_train is not None and len(selected) > max_train:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(selected), size=max_train, replace=False)
        selected = [selected[i] for i in sorted(idx.tolist())]

    n_target = len(selected)
    print(f"Loading dataset: N={n_target} samples (dataset_root={dataset_root})")

    names: list[str] = []
    ks: list[int] = []
    hm_feat: list[np.ndarray] = []
    imp_z: list[np.ndarray] = []
    occ_bin: list[np.ndarray] = []

    for i, (stem, k_cached) in enumerate(selected):
        hm_path = heatmap_dir / f"{stem}.npy"
        imp_path = imp_dir / f"{stem}.npy"
        occ_path = occ_dir / f"{stem}.npy"
        if not (hm_path.exists() and imp_path.exists() and occ_path.exists()):
            continue

        hm = np.load(hm_path)
        occ = _load_occ_bin(occ_path)
        imp0 = _load_impedance_channel0_z(imp_path)

        names.append(stem)
        occ_bin.append(occ)
        imp_z.append(imp0)
        hm_feat.append(_pool_heatmap_zscore(hm, hm_pool))
        ks.append(int(k_cached) if k_filter is not None else int(occ.sum()))

        if (i + 1) % 1000 == 0:
            print(f"Loaded {len(names)}/{n_target} dataset samples...")

    if not names:
        raise SystemExit("Dataset load resulted in 0 usable samples (missing files?)")

    return DatasetArrays(
        names=names,
        k_values=np.asarray(ks, dtype=np.int16),
        heatmap_feat=np.stack(hm_feat, axis=0).astype(np.float32, copy=False),
        imp_z=np.stack(imp_z, axis=0).astype(np.float32, copy=False),
        occ_bin=np.stack(occ_bin, axis=0).astype(np.uint8, copy=False),
    )


def _quantiles(x: np.ndarray) -> Quantiles:
    x = np.asarray(x)
    if x.size == 0:
        return {"min": None, "p10": None, "median": None, "p90": None, "max": None}
    q = np.quantile(x, [0.0, 0.1, 0.5, 0.9, 1.0])
    return {"min": float(q[0]), "p10": float(q[1]), "median": float(q[2]), "p90": float(q[3]), "max": float(q[4])}


def _percentiles_against_baseline(values: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Return percentile ranks in [0,1] for values relative to baseline.

    Percentile rank is computed as P(baseline <= value).
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    baseline = np.asarray(baseline, dtype=np.float64).reshape(-1)
    baseline = baseline[np.isfinite(baseline)]
    if baseline.size == 0 or values.size == 0:
        return np.asarray([], dtype=np.float64)
    b = np.sort(baseline)
    # right side: count of baseline <= value
    idx = np.searchsorted(b, values, side="right")
    return idx.astype(np.float64) / float(b.size)


def _fmt_q(d: Quantiles) -> str:
    def _one(k: str) -> str:
        v = d.get(k)
        return "NA" if v is None else f"{v:.4g}"

    return f"min={_one('min')} p10={_one('p10')} med={_one('median')} p90={_one('p90')} max={_one('max')}"


def score_generated_against_dataset(
    *,
    gen_dir: Path,
    dataset_root: Path,
    k_filter: int | None,
    hm_pool: int,
    baseline_n: int,
    seed: int,
    max_gen: int | None = None,
    max_train: int | None = None,
    out_csv: Path | None = None,
) -> ScoreSummary:
    """Compute novelty scores and optionally write a CSV (same format as CLI)."""
    if out_csv is None:
        out_csv = gen_dir / "novelty_report.csv"

    stats_path = dataset_root / "normalization_stats.json"
    if not stats_path.exists():
        raise SystemExit(f"Missing normalization stats: {stats_path}")
    stats = _load_json(stats_path)
    imp_log_mean = stats["Impedance"]["log_mean"]
    imp_log_std = stats["Impedance"]["log_std"]

    ds = load_dataset(
        dataset_root=dataset_root,
        hm_pool=hm_pool,
        max_train=max_train,
        k_filter=k_filter,
        seed=seed,
    )
    gen_names, gen_hm, gen_imp, gen_occ = load_generated(
        gen_dir=gen_dir,
        hm_pool=hm_pool,
        max_gen=max_gen,
        imp_log_mean=imp_log_mean,
        imp_log_std=imp_log_std,
    )

    n_train = len(ds.names)
    n_gen = len(gen_names)

    hm_mse = _mse_matrix(gen_hm, ds.heatmap_feat)
    imp_mse = _mse_matrix(gen_imp, ds.imp_z)
    occ_ham = _hamming_matrix(gen_occ, ds.occ_bin)
    combined = hm_mse + imp_mse + (occ_ham.astype(np.float32) / 52.0)

    nn_idx = np.argmin(combined, axis=1)
    nn_score = combined[np.arange(n_gen), nn_idx]
    nn_hm = hm_mse[np.arange(n_gen), nn_idx]
    nn_imp = imp_mse[np.arange(n_gen), nn_idx]
    nn_ham = occ_ham[np.arange(n_gen), nn_idx]

    nn_train_names = [ds.names[i] for i in nn_idx.tolist()]
    nn_train_k = ds.k_values[nn_idx]
    occ_exact = (nn_ham == 0)

    occ_stats = _occupancy_pattern_stats(ds.occ_bin, gen_occ)

    # Baseline: train→train nearest neighbor (combined)
    if n_train < 2:
        base_min = np.asarray([], dtype=np.float32)
    else:
        rng = np.random.default_rng(seed)
        baseline_n_eff = int(min(baseline_n, n_train))
        baseline_idx = rng.choice(n_train, size=baseline_n_eff, replace=False)
        base_hm = _mse_matrix(ds.heatmap_feat[baseline_idx], ds.heatmap_feat)
        base_imp = _mse_matrix(ds.imp_z[baseline_idx], ds.imp_z)
        base_ham = _hamming_matrix(ds.occ_bin[baseline_idx], ds.occ_bin)
        base_comb = base_hm + base_imp + (base_ham.astype(np.float32) / 52.0)
        base_comb[np.arange(baseline_n_eff), baseline_idx] = np.inf
        base_min = np.min(base_comb, axis=1)

    baseline_q = _quantiles(base_min)

    # Additional method A: percentile-based memorization score
    gen_percentiles = _percentiles_against_baseline(nn_score, base_min)
    gen_percentile_q: Quantiles | None
    mem_p01: float | None
    mem_p05: float | None
    mem_p10: float | None
    if gen_percentiles.size == n_gen and gen_percentiles.size > 0:
        gen_percentile_q = _quantiles(gen_percentiles)
        mem_p01 = float(np.mean(gen_percentiles <= 0.01))
        mem_p05 = float(np.mean(gen_percentiles <= 0.05))
        mem_p10 = float(np.mean(gen_percentiles <= 0.10))
    else:
        gen_percentile_q = None
        mem_p01 = None
        mem_p05 = None
        mem_p10 = None

    # Additional method B: training coverage (train→gen NN distances)
    train_to_gen_q: Quantiles | None
    train_cov_at_p90: float | None
    if n_train > 0 and n_gen > 0:
        train_to_gen_min = np.min(combined, axis=0)
        train_to_gen_q = _quantiles(train_to_gen_min)
        base_p90 = baseline_q["p90"]
        if base_p90 is None:
            train_cov_at_p90 = None
        else:
            train_cov_at_p90 = float(np.mean(train_to_gen_min <= float(base_p90)))
    else:
        train_to_gen_q = None
        train_cov_at_p90 = None

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "gen_sample",
                "nn_train_sample",
                "nn_train_K",
                "combined_score",
                "heatmap_mse",
                "impedance_mse",
                "occupancy_hamming",
                "occupancy_exact",
                "occ_pattern_in_train",
                "combined_percentile_vs_train_nn",
            ]
        )
        for i in range(n_gen):
            pct = ""
            if gen_percentiles.size == n_gen:
                pct = float(gen_percentiles[i])
            w.writerow(
                [
                    gen_names[i],
                    nn_train_names[i],
                    int(nn_train_k[i]),
                    float(nn_score[i]),
                    float(nn_hm[i]),
                    float(nn_imp[i]),
                    int(nn_ham[i]),
                    bool(occ_exact[i]),
                    bool(occ_stats["gen_in_train"][i]),
                    pct,
                ]
            )

    return ScoreSummary(
        n_train=n_train,
        n_gen=n_gen,
        k_filter=k_filter,
        baseline_combined=baseline_q,
        gen_to_train_combined=_quantiles(nn_score),
        gen_to_train_heatmap_mse=_quantiles(nn_hm),
        gen_to_train_impedance_mse=_quantiles(nn_imp),
        gen_to_train_occupancy_hamming=_quantiles(nn_ham.astype(np.float32)),
        occupancy_exact_count=int(occ_exact.sum()),
        train_occ_unique=int(occ_stats["train_unique"]),
        gen_occ_unique=int(occ_stats["gen_unique"]),
        gen_occ_new_unique=int(occ_stats["gen_new_unique"]),
        gen_occ_new_rate=float(occ_stats["gen_new_rate"]),
        gen_occ_mode_frac=float(occ_stats["gen_mode_frac"]),
        occ_jsd=float(occ_stats["jsd"]),
        gen_to_train_combined_percentile=gen_percentile_q,
        mem_suspect_rate_p01=mem_p01,
        mem_suspect_rate_p05=mem_p05,
        mem_suspect_rate_p10=mem_p10,
        train_to_gen_combined=train_to_gen_q,
        train_to_gen_coverage_at_base_p90=train_cov_at_p90,
    )


def load_generated(
    *,
    gen_dir: Path,
    hm_pool: int,
    max_gen: Optional[int],
    imp_log_mean: float,
    imp_log_std: float,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    sample_dirs = _iter_sample_dirs(gen_dir)
    if not sample_dirs:
        raise SystemExit(f"No data_sample_* directories found in {gen_dir}")

    if max_gen is not None:
        sample_dirs = sample_dirs[:max_gen]

    names: list[str] = []
    hm_feat = []
    imp_z = []
    occ_bin = []

    for d in sample_dirs:
        hm_path = d / "heatmap_zscore.npy"
        occ_path = d / "occupancy_map.npy"
        imp_path = d / "impedance_profile.npy"
        if not (hm_path.exists() and occ_path.exists() and imp_path.exists()):
            raise SystemExit(f"Missing required generated files in {d} (need heatmap_zscore.npy, occupancy_map.npy, impedance_profile.npy)")

        hm = np.load(hm_path)
        occ = _load_occ_bin(occ_path)
        imp_log = np.load(imp_path)

        names.append(d.name)
        hm_feat.append(_pool_heatmap_zscore(hm, hm_pool))
        imp_z.append(_imp_log_to_z(imp_log, log_mean=imp_log_mean, log_std=imp_log_std))
        occ_bin.append(occ)

    return (
        names,
        np.stack(hm_feat, axis=0).astype(np.float32, copy=False),
        np.stack(imp_z, axis=0).astype(np.float32, copy=False),
        np.stack(occ_bin, axis=0).astype(np.uint8, copy=False),
    )


def _describe_dist(x: np.ndarray) -> str:
    if x.size == 0:
        return "(empty)"
    q = np.quantile(x, [0.0, 0.1, 0.5, 0.9, 1.0])
    return f"min={q[0]:.4g} p10={q[1]:.4g} med={q[2]:.4g} p90={q[3]:.4g} max={q[4]:.4g}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gen-dir",
        type=Path,
        default=Path("evaluation/novelty/runs/K5_noveltyN50"),
        help="Folder containing data_sample_* (e.g. evaluation/novelty/runs/K5_noveltyN50)",
    )
    ap.add_argument("--dataset-root", type=Path, default=Path("datasets/data_norm"), help="Dataset root with heatmap/Imp/Occ_map")
    ap.add_argument("--K", type=int, default=None, help="If set, filter dataset to this K (recommended). If omitted, inferred from --gen-dir when possible.")
    ap.add_argument("--max-gen", type=int, default=None, help="Limit number of generated samples")
    ap.add_argument("--max-train", type=int, default=None, help="Subsample training set (after K filtering)")
    ap.add_argument("--hm-pool", type=int, default=16, help="Heatmap mean-pool size (e.g. 8, 16, 32). Must divide 64.")
    ap.add_argument("--baseline-n", type=int, default=200, help="How many train samples to use for train→train NN baseline")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-csv", type=Path, default=None, help="Output CSV path (default: <gen-dir>/novelty_report.csv)")
    args = ap.parse_args()

    gen_dir: Path = args.gen_dir
    dataset_root: Path = args.dataset_root

    if args.out_csv is None:
        out_csv = gen_dir / "novelty_report.csv"
    else:
        out_csv = args.out_csv

    k_filter = args.K
    if k_filter is None:
        k_filter = _infer_k_from_path(gen_dir)

    print(f"gen-dir:       {gen_dir}")
    print(f"dataset-root:  {dataset_root}")
    print(f"K filter:      {k_filter if k_filter is not None else '(none)'}")
    print(f"hm-pool:       {args.hm_pool}x{args.hm_pool}")

    summary = score_generated_against_dataset(
        gen_dir=gen_dir,
        dataset_root=dataset_root,
        k_filter=k_filter,
        hm_pool=args.hm_pool,
        baseline_n=args.baseline_n,
        seed=args.seed,
        max_gen=args.max_gen,
        max_train=args.max_train,
        out_csv=out_csv,
    )

    print(f"Train N={summary.n_train} | Generated M={summary.n_gen}")
    print("\nNearest-neighbor distance summary")
    print(f"  train→train baseline (combined): {_fmt_q(summary.baseline_combined)}")
    # Keep the original pretty-print for gen→train using actual arrays by reusing the dicts.
    print(
        "  gen→train (combined):           "
        f"{_fmt_q(summary.gen_to_train_combined)}"
    )
    print(
        "  gen→train heatmap MSE:          "
        f"{_fmt_q(summary.gen_to_train_heatmap_mse)}"
    )
    print(
        "  gen→train impedance MSE:        "
        f"{_fmt_q(summary.gen_to_train_impedance_mse)}"
    )
    ham = summary.gen_to_train_occupancy_hamming
    ham_min = ham["min"]
    ham_med = ham["median"]
    ham_max = ham["max"]
    if ham_min is None or ham_med is None or ham_max is None:
        ham_str = "min=NA med=NA max=NA"
    else:
        ham_str = f"min={int(ham_min)} med={int(ham_med)} max={int(ham_max)}"
    print("  gen→train occupancy hamming:    " + ham_str)
    print(f"  gen→train occupancy exact:      {summary.occupancy_exact_count}/{summary.n_gen}")

    if summary.gen_occ_new_rate is not None:
        print("\nOccupancy-pattern novelty (exact patterns)")
        print(f"  train unique patterns:          {summary.train_occ_unique}")
        print(f"  gen unique patterns:            {summary.gen_occ_unique}")
        print(f"  gen new unique patterns:        {summary.gen_occ_new_unique}")
        print(f"  gen new-pattern rate:           {summary.gen_occ_new_rate:.4g}")
        print(f"  gen occupancy mode fraction:    {summary.gen_occ_mode_frac:.4g}")
        print(f"  occupancy JSD (train vs gen):   {summary.occ_jsd:.4g}")

    if summary.gen_to_train_combined_percentile is not None:
        p = summary.gen_to_train_combined_percentile
        print("\nMemorization percentile test (combined NN distance)")
        print("  Interpreting percentiles: lower = unusually close to training")
        print(f"  gen→train NN percentile:        {_fmt_q(p)}")
        if summary.mem_suspect_rate_p01 is not None:
            print(f"  suspect rate (<=1%):            {summary.mem_suspect_rate_p01:.4g}")
            print(f"  suspect rate (<=5%):            {summary.mem_suspect_rate_p05:.4g}")
            print(f"  suspect rate (<=10%):           {summary.mem_suspect_rate_p10:.4g}")

    if summary.train_to_gen_combined is not None:
        print("\nTrain coverage (train→gen NN distance)")
        print(f"  train→gen (combined):           {_fmt_q(summary.train_to_gen_combined)}")
        if summary.train_to_gen_coverage_at_base_p90 is not None:
            print(f"  coverage @ baseline p90:        {summary.train_to_gen_coverage_at_base_p90:.4g}")
    print(f"\nWrote: {out_csv}")


if __name__ == "__main__":
    main()
