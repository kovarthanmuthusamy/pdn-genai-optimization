#!/usr/bin/env python3
"""Evaluate a trained VAE checkpoint (reconstruction + latent diagnostics).

This script is intentionally lightweight:
- No extra dependencies beyond numpy/torch/matplotlib
- Writes a compact Markdown report with a few plots
- Writes a CSV with per-sample metrics for deeper inspection

Default target checkpoint:
  experiments/exp027_sigma_reg_tuning/checkpoints/checkpoint_epoch_400.pt

Run:
  python evaluation/vae/run_vae_eval.py
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, cast

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EvalConfig:
    checkpoint: Path = Path("experiments/exp027_sigma_reg_tuning/checkpoints/checkpoint_epoch_400.pt")
    dataset_root: Path = Path("datasets/data_eval_norm")
    out_dir: Path = Path("evaluation/vae")

    n_eval: int = 2048
    batch_size: int = 64
    n_gen: int = 2048
    shared_temp: float = 1.5

    seed: int = 0
    force_cpu: bool = False
    active_var_thresh: float = 1e-2


# =====================
# CONFIG (edit here)
# =====================
CONFIG = EvalConfig()


class VAEModel(Protocol):
    def to(self, device: torch.device) -> VAEModel: ...

    def eval(self) -> VAEModel: ...

    def __call__(
        self,
        heatmap: torch.Tensor,
        occupancy: torch.Tensor,
        impedance: torch.Tensor,
        K: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]: ...

    def decode(self, z: torch.Tensor, K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def encode_cross_modal(
        self,
        source: str,
        K: torch.Tensor,
        *,
        heatmap: torch.Tensor | None = None,
        impedance: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


def _find_project_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "experiments").is_dir() and (p / "datasets").is_dir():
            return p
    return start


def _parse_sample_id(name: str) -> int | None:
    m = re.fullmatch(r"sample_(\d+)\.npy", name)
    if not m:
        return None
    return int(m.group(1))


@dataclass(frozen=True)
class LoadedModel:
    model: VAEModel
    ckpt_config: dict
    latent_stats: dict | None
    per_k_latent_stats: dict | None
    epoch: int | None


def _load_model(*, project_root: Path, checkpoint_path: Path, device: torch.device) -> LoadedModel:
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from experiments.exp026_change_with_latent.codes.vae_multi_input_simple import (  # noqa: WPS433
        MultiInputVAE,
    )

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})

    latent_dim = int(cfg.get("latent_dim", 32))
    cond_dim = int(cfg.get("cond_dim", 8))

    model = MultiInputVAE(latent_dim=latent_dim, cond_dim=cond_dim)
    model_state = model.state_dict()

    ckpt_state = ckpt["model_state_dict"]
    compat = {k: v for (k, v) in ckpt_state.items() if k in model_state and model_state[k].shape == v.shape}
    load_result = model.load_state_dict(compat, strict=False)

    required_prefixes = (
        "heatmap_fc.",
        "heatmap_dec_deconv1.",
        "heatmap_dec_deconv2.",
        "occupancy_decoder.",
        "impedance_decoder.",
        "heatmap_mu.",
        "occupancy_mu.",
        "impedance_mu.",
    )
    missing = [k for k in load_result.missing_keys if k.startswith(required_prefixes)]
    if missing:
        raise RuntimeError(f"Missing generation-critical weights: {missing[:12]}")

    model.to(device).eval()

    return LoadedModel(
        model=cast(VAEModel, model),
        ckpt_config=cfg,
        latent_stats=ckpt.get("latent_stats"),
        per_k_latent_stats=ckpt.get("per_K_latent_stats"),
        epoch=int(ckpt.get("epoch")) if "epoch" in ckpt else None,
    )


def _load_ids(*, occ_dir: Path) -> list[int]:
    ids: list[int] = []
    for p in occ_dir.glob("sample_*.npy"):
        sid = _parse_sample_id(p.name)
        if sid is not None:
            ids.append(sid)
    ids.sort()
    if not ids:
        raise SystemExit(f"No sample_*.npy files found in {occ_dir}")
    return ids


def _iter_batches(ids: list[int], *, batch_size: int) -> Iterable[list[int]]:
    for i in range(0, len(ids), batch_size):
        yield ids[i : i + batch_size]


def _load_batch(*, dataset_root: Path, batch_ids: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hm_dir = dataset_root / "heatmap"
    imp_dir = dataset_root / "Imp"
    occ_dir = dataset_root / "Occ_map"

    hm_list = []
    imp_list = []
    occ_list = []

    for sid in batch_ids:
        hm_list.append(np.load(hm_dir / f"sample_{sid}.npy"))
        imp_list.append(np.load(imp_dir / f"sample_{sid}.npy"))
        occ_list.append(np.load(occ_dir / f"sample_{sid}.npy"))

    hm = np.stack(hm_list, axis=0)
    imp = np.stack(imp_list, axis=0)
    occ = np.stack(occ_list, axis=0)

    return hm, occ, imp


def _topk_match_rate(probs: torch.Tensor, occ_true: torch.Tensor) -> torch.Tensor:
    """Top-K set match rate, where K is taken from the target occupancy count.

    Returns (B,) with values in [0, 1]. For K=0, returns 1.0.
    """

    if probs.ndim != 2 or occ_true.ndim != 2 or probs.shape != occ_true.shape:
        raise ValueError(f"Expected probs and occ_true as (B,52), got {probs.shape} and {occ_true.shape}")

    bsz, d = probs.shape
    k_true = (occ_true > 0.5).sum(dim=1).to(torch.int64)

    out = torch.empty((bsz,), dtype=torch.float32, device=probs.device)
    for i in range(bsz):
        k = int(k_true[i].item())
        if k <= 0:
            out[i] = 1.0
            continue
        idx = torch.topk(probs[i], k).indices
        out[i] = (occ_true[i, idx] > 0.5).to(torch.float32).mean()
    return out


def _plot_recon_vs_k(*, k: np.ndarray, hm_mse: np.ndarray, imp_mse: np.ndarray, occ_bce: np.ndarray, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    ks = np.unique(k)
    hm_by_k = np.asarray([np.nanmean(hm_mse[k == kk]) for kk in ks], dtype=np.float64)
    imp_by_k = np.asarray([np.nanmean(imp_mse[k == kk]) for kk in ks], dtype=np.float64)
    bce_by_k = np.asarray([np.nanmean(occ_bce[k == kk]) for kk in ks], dtype=np.float64)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 9), sharex=True)

    axes[0].plot(ks, hm_by_k, marker="o", markersize=3, linewidth=1.5)
    axes[0].set_ylabel("heatmap MSE")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(ks, imp_by_k, marker="o", markersize=3, linewidth=1.5)
    axes[1].set_ylabel("impedance MSE")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(ks, bce_by_k, marker="o", markersize=3, linewidth=1.5)
    axes[2].set_ylabel("occupancy BCE")
    axes[2].set_xlabel("K")
    axes[2].grid(True, alpha=0.25)

    fig.suptitle("Reconstruction error vs K (mean)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_kl_per_dim(*, kl_per_dim: np.ndarray, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    x = np.asarray(kl_per_dim, dtype=np.float64)
    order = np.argsort(-x)
    x_sorted = x[order]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(x_sorted)), x_sorted, width=0.9)
    ax.set_title("Mean KL per latent dim (sorted)")
    ax.set_xlabel("latent dim (sorted)")
    ax.set_ylabel("KL")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_occ_rates(*, occ_real_rate: np.ndarray, occ_gen_rate: np.ndarray, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    r = np.asarray(occ_real_rate, dtype=np.float64).reshape(-1)
    g = np.asarray(occ_gen_rate, dtype=np.float64).reshape(-1)
    if r.shape != g.shape:
        raise ValueError("Real/gen occupancy rate shapes do not match")

    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(r.size)
    ax.plot(x, r, label="real", linewidth=1.5)
    ax.plot(x, g, label="generated", linewidth=1.5)
    ax.set_title("Occupancy rate per slot (real vs generated)")
    ax.set_xlabel("slot index (0..51)")
    ax.set_ylabel("P(occupied)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


_K_BUCKETS: tuple[tuple[int, int, str], ...] = (
    (0, 13, "K 0-12"),
    (13, 26, "K 13-25"),
    (26, 39, "K 26-38"),
    (39, 53, "K 39-52"),
)


def _pca_from_mu(mu: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute PCA from covariance (no sklearn).

    Returns:
        mean: (D,)
        components: (D, D) rows are PCs (like sklearn components_)
        explained_variance: (D,)
        explained_variance_ratio: (D,)
    """

    x = np.asarray(mu, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected mu as (N,D), got {x.shape}")

    n, d = x.shape
    mean = x.mean(axis=0)
    if n < 2:
        components = np.eye(d, dtype=np.float64)
        ev = np.zeros((d,), dtype=np.float64)
        evr = np.zeros((d,), dtype=np.float64)
        return mean, components, ev, evr

    xc = x - mean
    cov = (xc.T @ xc) / float(n - 1)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]  # columns

    components = eigvecs.T  # rows
    total = float(np.sum(eigvals))
    if not np.isfinite(total) or total <= 0.0:
        evr = np.zeros_like(eigvals)
    else:
        evr = eigvals / total

    return mean, components, eigvals, evr


def _nan_to_none_list(x: np.ndarray) -> list[float | None]:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    return [None if not np.isfinite(v) else float(v) for v in arr.tolist()]


def _plot_latent_pca_scatter(
    *,
    mu: np.ndarray,
    K: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    evr: np.ndarray,
    out_path: Path,
    max_points: int = 30000,
    seed: int = 0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433
    from matplotlib.colors import Normalize  # noqa: WPS433

    x = np.asarray(mu, dtype=np.float64)
    k = np.asarray(K, dtype=np.int64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != k.shape[0]:
        raise ValueError(f"mu/K shape mismatch: {x.shape} vs {k.shape}")

    n = x.shape[0]
    idx = np.arange(n)
    if n > int(max_points):
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(idx, size=int(max_points), replace=False)

    xc = x[idx] - mean
    emb2 = xc @ components[:2].T

    fig, ax = plt.subplots(figsize=(9, 7))
    norm = Normalize(vmin=0, vmax=52)
    sc = ax.scatter(
        emb2[:, 0],
        emb2[:, 1],
        c=k[idx],
        cmap="plasma",
        norm=norm,
        s=8,
        alpha=0.65,
        linewidths=0,
    )
    plt.colorbar(sc, ax=ax, label="K (occupied slots)")
    ax.set_title("PCA of fused posterior means (PC1 vs PC2)")
    if evr.size >= 2:
        ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}% var)")
    else:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_latent_pca_variance(
    *,
    evr: np.ndarray,
    dims_90: int,
    dims_95: int,
    dims_99: int,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    evr = np.asarray(evr, dtype=np.float64).reshape(-1)
    cumev = np.cumsum(evr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    top = int(min(30, evr.size))
    if top > 0:
        ax1.bar(range(1, top + 1), evr[:top] * 100.0, color="steelblue", alpha=0.8)
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Explained variance (%)")
    ax1.set_title("Per-component variance (top 30)")
    ax1.grid(True, axis="y", alpha=0.35)

    ax2.plot(range(1, len(cumev) + 1), cumev * 100.0, color="steelblue", lw=2)
    for thr, d, col in [(90, dims_90, "gold"), (95, dims_95, "orange"), (99, dims_99, "tomato")]:
        if d > 0:
            ax2.axhline(thr, ls="--", lw=1, color=col, alpha=0.7)
            ax2.axvline(d, ls="--", lw=1, color=col, alpha=0.7, label=f"{thr}% @ dim {d}")
    ax2.set_xlabel("Number of components")
    ax2.set_ylabel("Cumulative variance (%)")
    ax2.set_title("Cumulative explained variance")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.35)
    ax2.set_xlim(0, int(min(evr.size, 60)))

    fig.suptitle("Latent space dimensionality usage", fontsize=13, fontweight="bold")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_latent_per_dim_mu_sigma(
    *,
    mu_std: np.ndarray,
    sigma_mean: np.ndarray,
    out_path: Path,
    dead_sigma_thresh: float = 0.95,
    dead_mu_std_thresh: float = 0.05,
) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    mu_std = np.asarray(mu_std, dtype=np.float64).reshape(-1)
    sigma_mean = np.asarray(sigma_mean, dtype=np.float64).reshape(-1)
    if mu_std.shape != sigma_mean.shape:
        raise ValueError("mu_std and sigma_mean must have same shape")

    order = np.argsort(-mu_std)
    dead_mask = (sigma_mean > float(dead_sigma_thresh)) & (mu_std < float(dead_mu_std_thresh))
    dead_dims = int(np.sum(dead_mask))

    x = np.arange(mu_std.size)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.bar(x, mu_std[order], color="#4C72B0", alpha=0.85, width=1.0)
    ax1.axhline(float(dead_mu_std_thresh), ls="--", color="tomato", lw=1, label=f"dead μ-std thresh ({dead_mu_std_thresh})")
    ax1.set_ylabel("μ std across eval set")
    ax1.set_title("Per-dim posterior mean spread (sorted)")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.bar(x, sigma_mean[order], color="#55A868", alpha=0.85, width=1.0)
    ax2.axhline(1.0, ls="--", color="grey", lw=1, label="prior σ=1")
    ax2.axhline(0.45, ls="--", color="orange", lw=1.2, label="σ target≈0.45")
    ax2.axhline(float(dead_sigma_thresh), ls=":", color="tomato", lw=1, label=f"dead σ thresh ({dead_sigma_thresh})")
    ax2.set_ylabel("Mean posterior σ")
    ax2.set_xlabel("Latent dim (sorted by μ std)")
    ax2.set_title("Per-dim mean posterior σ (same order)")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout(h_pad=2.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    return dead_dims


def _plot_hist_by_k_bucket(
    *,
    values: np.ndarray,
    K: np.ndarray,
    out_path: Path,
    xlabel: str,
    title: str,
    target_vlines: list[tuple[float, str, str, str]] | None = None,
) -> list[dict]:
    """Histogram values by K buckets.

    target_vlines: list of (x, linestyle, color, label)
    Returns: list of bucket stats dicts.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    v = np.asarray(values, dtype=np.float64).reshape(-1)
    k = np.asarray(K, dtype=np.int64).reshape(-1)
    if v.shape[0] != k.shape[0]:
        raise ValueError(f"values/K shape mismatch: {v.shape} vs {k.shape}")

    fig, axes = plt.subplots(1, len(_K_BUCKETS), figsize=(14, 4), sharey=True)
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    stats: list[dict] = []
    for ax, (lo, hi, label), col in zip(axes, _K_BUCKETS, colors):
        mask = (k >= int(lo)) & (k < int(hi))
        vals = v[mask]
        if vals.size == 0:
            ax.set_title(f"{label}\n(no samples)")
            stats.append({"label": label, "lo": int(lo), "hi": int(hi), "n": 0})
            continue

        mean = float(np.mean(vals))
        std = float(np.std(vals))
        ax.hist(vals, bins=35, color=col, alpha=0.85, edgecolor="white", density=True)
        ax.axvline(mean, color="black", ls="-", lw=1.5, label=f"mean={mean:.3g}")
        if target_vlines:
            for x0, ls, c, lbl in target_vlines:
                ax.axvline(float(x0), ls=ls, color=c, lw=1.2, label=lbl)

        ax.set_title(f"{label} (n={int(mask.sum())})")
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

        stats.append({"label": label, "lo": int(lo), "hi": int(hi), "n": int(mask.sum()), "mean": mean, "std": std})

    axes[0].set_ylabel("Density")
    overall_mean = float(np.mean(v))
    fig.suptitle(f"{title} (overall mean={overall_mean:.3g})", fontsize=12, fontweight="bold")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    return stats


def _linear_probe_k_from_mu(mu: np.ndarray, K: np.ndarray, *, seed: int = 0) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Fit a linear regression K<-mu and report (R², MAE) on a random 80/20 split."""

    x = np.asarray(mu, dtype=np.float64)
    y = np.asarray(K, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError(f"mu/K shape mismatch: {x.shape} vs {y.shape}")

    n = x.shape[0]
    if n < 10:
        return float("nan"), float("nan"), y, np.full_like(y, np.nan)

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    split = max(1, int(0.8 * n))
    tr = perm[:split]
    te = perm[split:]

    x_tr = x[tr]
    y_tr = y[tr]
    x_te = x[te]
    y_te = y[te]

    x_tr_aug = np.concatenate([x_tr, np.ones((x_tr.shape[0], 1), dtype=np.float64)], axis=1)
    x_te_aug = np.concatenate([x_te, np.ones((x_te.shape[0], 1), dtype=np.float64)], axis=1)

    w, *_ = np.linalg.lstsq(x_tr_aug, y_tr, rcond=None)
    pred = x_te_aug @ w

    sse = float(np.sum((pred - y_te) ** 2))
    sst = float(np.sum((y_te - float(np.mean(y_te))) ** 2))
    r2 = float("nan") if sst <= 0.0 else float(1.0 - sse / sst)
    mae = float(np.mean(np.abs(pred - y_te)))
    return r2, mae, y_te, pred


def _plot_probe_k_from_mu(*, y_true: np.ndarray, y_pred: np.ndarray, r2: float, mae: float, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if yt.size < 10 or yp.size < 10:
        return

    mask = np.isfinite(yt) & np.isfinite(yp)
    if int(np.sum(mask)) < 2:
        return
    yt = yt[mask]
    yp = yp[mask]

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(yt, yp, s=10, alpha=0.35, linewidths=0)
    lo = float(min(np.min(yt), np.min(yp)))
    hi = float(max(np.max(yt), np.max(yp)))
    ax.plot([lo, hi], [lo, hi], color="black", lw=1)
    ax.set_title(f"Linear probe: predict K from fused μ\nR²={r2:.3f}  MAE={mae:.2f}")
    ax.set_xlabel("True K")
    ax.set_ylabel("Predicted K")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_expert_disagreement_by_k(
    *,
    K_counts: np.ndarray,
    per_modality: dict[str, dict[str, np.ndarray]],
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    Ks = np.arange(int(K_counts.size))
    modalities = ("heatmap", "occupancy", "impedance")

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    fig.suptitle("Modality experts vs fused posterior (binned by K)", fontsize=13, fontweight="bold")

    for row, m in enumerate(modalities):
        if m not in per_modality:
            continue

        mean_diff = np.asarray(per_modality[m]["mu_rms_diff_mean"], dtype=np.float64)
        std_diff = np.asarray(per_modality[m]["mu_rms_diff_std"], dtype=np.float64)
        mean_sig = np.asarray(per_modality[m]["sigma_mean"], dtype=np.float64)
        std_sig = np.asarray(per_modality[m]["sigma_std"], dtype=np.float64)

        mask = np.isfinite(mean_diff) & np.isfinite(std_diff)
        axd = axes[row, 0]
        axd.plot(Ks[mask], mean_diff[mask], color="#4C72B0", lw=2)
        axd.fill_between(
            Ks[mask],
            (mean_diff - std_diff)[mask],
            (mean_diff + std_diff)[mask],
            color="#4C72B0",
            alpha=0.2,
        )
        axd.set_ylabel(f"{m}\nμ RMS diff")
        axd.grid(True, alpha=0.25)

        mask2 = np.isfinite(mean_sig) & np.isfinite(std_sig)
        axs = axes[row, 1]
        axs.plot(Ks[mask2], mean_sig[mask2], color="#55A868", lw=2)
        axs.fill_between(
            Ks[mask2],
            (mean_sig - std_sig)[mask2],
            (mean_sig + std_sig)[mask2],
            color="#55A868",
            alpha=0.2,
        )
        axs.axhline(0.45, ls="--", color="orange", lw=1.2)
        axs.set_ylabel(f"{m}\nmean σ")
        axs.grid(True, alpha=0.25)

    axes[2, 0].set_xlabel("K")
    axes[2, 1].set_xlabel("K")
    axes[0, 0].set_title("Expert vs fused μ disagreement (mean±std)")
    axes[0, 1].set_title("Expert uncertainty σ (mean±std)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # type: ignore

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _fmt(x: float) -> str:
    if not np.isfinite(x):
        return "NA"
    return f"{x:.4g}"


def _extract_latent_mu_std(*, latent_stats: dict | None, latent_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (mu_per_dim, agg_std_per_dim) in numpy float32."""

    if not latent_stats:
        return np.zeros((latent_dim,), dtype=np.float32), np.ones((latent_dim,), dtype=np.float32)

    s = latent_stats.get("latent", {})
    if "mu_mean_per_dim" in s and "agg_std_per_dim" in s:
        mu = np.asarray(s["mu_mean_per_dim"], dtype=np.float32).reshape(-1)
        std = np.asarray(s["agg_std_per_dim"], dtype=np.float32).reshape(-1)
        if mu.size != latent_dim or std.size != latent_dim:
            raise ValueError("latent_stats per-dim arrays do not match latent_dim")
        return mu, std

    mu_val = float(s.get("mu_mean", 0.0))
    agg_std = float((s.get("mu_std", 1.0) ** 2 + s.get("sigma_mean", 1.0) ** 2) ** 0.5)
    return np.full((latent_dim,), mu_val, dtype=np.float32), np.full((latent_dim,), agg_std, dtype=np.float32)


def _build_per_k_latent_tables(
    *,
    per_k_latent_stats: dict | None,
    global_mu: np.ndarray,
    global_std: np.ndarray,
    latent_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build lookup tables for K=0..52.

    Missing K values fall back to the nearest available K; if no per-K stats exist,
    uses global stats everywhere.
    """

    mu_table = np.tile(global_mu.reshape(1, -1), (53, 1)).astype(np.float32, copy=False)
    std_table = np.tile(global_std.reshape(1, -1), (53, 1)).astype(np.float32, copy=False)

    if not per_k_latent_stats:
        return mu_table, std_table

    available = sorted(int(k) for k in per_k_latent_stats.keys() if str(k).isdigit())
    if not available:
        return mu_table, std_table

    for k in range(53):
        key = str(k)
        if key in per_k_latent_stats:
            stats_k = per_k_latent_stats[key]
        else:
            nearest = min(available, key=lambda x: abs(x - k))
            stats_k = per_k_latent_stats[str(nearest)]

        s = stats_k.get("latent", {})
        if "mu_mean_per_dim" in s and "agg_std_per_dim" in s:
            mu_k = np.asarray(s["mu_mean_per_dim"], dtype=np.float32).reshape(-1)
            std_k = np.asarray(s["agg_std_per_dim"], dtype=np.float32).reshape(-1)
            if mu_k.size == latent_dim and std_k.size == latent_dim:
                mu_table[k] = mu_k
                std_table[k] = std_k

    return mu_table, std_table



def run_eval(cfg: EvalConfig) -> None:
    project_root = _find_project_root(Path(__file__))
    os.chdir(project_root)

    device = torch.device("cpu" if cfg.force_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    out_dir: Path = cfg.out_dir
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {cfg.checkpoint}")
    loaded = _load_model(project_root=project_root, checkpoint_path=cfg.checkpoint, device=device)
    model = loaded.model

    dataset_root: Path = cfg.dataset_root
    occ_dir = dataset_root / "Occ_map"
    ids = _load_ids(occ_dir=occ_dir)

    rng = np.random.default_rng(int(cfg.seed))
    n_eval = min(int(cfg.n_eval), len(ids))
    eval_ids = rng.choice(np.asarray(ids, dtype=np.int64), size=n_eval, replace=False).tolist()

    # Storage for per-sample stats
    k_all: list[int] = []
    hm_mse_all: list[float] = []
    hm_mae_all: list[float] = []
    imp_mse_all: list[float] = []
    imp_mae_all: list[float] = []
    occ_bce_all: list[float] = []
    occ_match_all: list[float] = []
    occ_exact_all: list[float] = []
    kl_total_all: list[float] = []

    # Store fused posterior stats per-sample (for latent analysis plots)
    mu_batches: list[np.ndarray] = []
    logvar_batches: list[np.ndarray] = []
    fused_sigma_mean_per_sample: list[float] = []

    # Expert-vs-fused disagreement (binned by exact K)
    k_counts = np.zeros((53,), dtype=np.int64)
    expert_acc: dict[str, dict[str, np.ndarray]] = {
        m: {
            "mu_rms_diff_sum": np.zeros((53,), dtype=np.float64),
            "mu_rms_diff_sumsq": np.zeros((53,), dtype=np.float64),
            "sigma_sum": np.zeros((53,), dtype=np.float64),
            "sigma_sumsq": np.zeros((53,), dtype=np.float64),
        }
        for m in ("heatmap", "occupancy", "impedance")
    }

    # Cross-modal recon (heatmap-only / impedance-only)
    hm_only_occ_bce: list[float] = []
    hm_only_imp_mse: list[float] = []
    imp_only_occ_bce: list[float] = []
    imp_only_hm_mse: list[float] = []

    # Latent aggregates
    cfg_latent_dim = int(loaded.ckpt_config.get("latent_dim", 32))
    mu_sum = torch.zeros((cfg_latent_dim,), device=device)
    mu_sumsq = torch.zeros((cfg_latent_dim,), device=device)
    sigma_sum = torch.zeros((cfg_latent_dim,), device=device)
    kl_dim_sum = torch.zeros((cfg_latent_dim,), device=device)
    latent_n = 0

    real_occ_sum = torch.zeros((52,), device=device)
    real_hm_sum = 0.0
    real_hm_sumsq = 0.0
    real_imp_sum = 0.0
    real_imp_sumsq = 0.0
    real_hm_n = 0
    real_imp_n = 0

    bsz = int(cfg.batch_size)

    print(f"Evaluating recon/latent on {len(eval_ids)} samples (batch={bsz}) on {device}...")
    model.eval()
    with torch.no_grad():
        for batch_ids in _iter_batches(eval_ids, batch_size=bsz):
            hm_np, occ_np, imp_np = _load_batch(dataset_root=dataset_root, batch_ids=batch_ids)

            hm = torch.from_numpy(hm_np).to(device=device, dtype=torch.float32)
            occ = torch.from_numpy(occ_np).to(device=device, dtype=torch.float32)
            imp = torch.from_numpy(imp_np).to(device=device, dtype=torch.float32)

            k = (occ > 0.5).sum(dim=1).to(torch.int64)

            hm_pred, occ_logits, imp_pred, mu, logvar, expert_stats = model(hm, occ, imp, k)

            mu_batches.append(mu.detach().cpu().numpy())
            logvar_batches.append(logvar.detach().cpu().numpy())
            fused_sigma_mean_per_sample.extend(torch.exp(0.5 * logvar).mean(dim=1).detach().cpu().tolist())

            # Expert-vs-fused disagreement stats (uses expert_stats from the model forward)
            k_cpu = k.detach().cpu().numpy().astype(np.int64, copy=False)
            k_counts += np.bincount(k_cpu, minlength=53)
            if isinstance(expert_stats, dict):
                for m in ("heatmap", "occupancy", "impedance"):
                    if m not in expert_stats:
                        continue
                    m_mu, m_lv = expert_stats[m]
                    mu_rms_diff = torch.sqrt(((m_mu - mu) ** 2).mean(dim=1))
                    sig_mean = torch.exp(0.5 * m_lv).mean(dim=1)

                    diff_np = mu_rms_diff.detach().cpu().numpy().astype(np.float64, copy=False)
                    sig_np = sig_mean.detach().cpu().numpy().astype(np.float64, copy=False)

                    expert_acc[m]["mu_rms_diff_sum"] += np.bincount(k_cpu, weights=diff_np, minlength=53)
                    expert_acc[m]["mu_rms_diff_sumsq"] += np.bincount(k_cpu, weights=diff_np**2, minlength=53)
                    expert_acc[m]["sigma_sum"] += np.bincount(k_cpu, weights=sig_np, minlength=53)
                    expert_acc[m]["sigma_sumsq"] += np.bincount(k_cpu, weights=sig_np**2, minlength=53)

            hm_mse = ((hm_pred - hm) ** 2).mean(dim=(1, 2, 3))
            hm_mae = (hm_pred - hm).abs().mean(dim=(1, 2, 3))
            imp_mse = ((imp_pred - imp) ** 2).mean(dim=(1, 2))
            imp_mae = (imp_pred - imp).abs().mean(dim=(1, 2))

            occ_bce = F.binary_cross_entropy_with_logits(occ_logits, occ, reduction="none").mean(dim=1)
            occ_match = _topk_match_rate(torch.sigmoid(occ_logits), occ)
            occ_exact = (occ_match >= 0.999999).to(torch.float32)

            kl_per_dim = -0.5 * (1.0 + logvar - mu**2 - torch.exp(logvar))
            kl_total = kl_per_dim.sum(dim=1)

            # Cross-modal: heatmap-only → recon others
            z_hm = model.encode_cross_modal("heatmap", k, heatmap=hm)
            hm_pred_hm, occ_logits_hm, imp_pred_hm = model.decode(z_hm, k)
            hm_only_occ_bce_batch = F.binary_cross_entropy_with_logits(occ_logits_hm, occ, reduction="none").mean(dim=1)
            hm_only_imp_mse_batch = ((imp_pred_hm - imp) ** 2).mean(dim=(1, 2))

            # Cross-modal: impedance-only → recon others
            z_imp = model.encode_cross_modal("impedance", k, impedance=imp)
            hm_pred_imp, occ_logits_imp, _imp_pred_imp = model.decode(z_imp, k)
            imp_only_occ_bce_batch = F.binary_cross_entropy_with_logits(occ_logits_imp, occ, reduction="none").mean(dim=1)
            imp_only_hm_mse_batch = ((hm_pred_imp - hm) ** 2).mean(dim=(1, 2, 3))

            # Accumulate
            k_all.extend(k.detach().cpu().tolist())
            hm_mse_all.extend(hm_mse.detach().cpu().tolist())
            hm_mae_all.extend(hm_mae.detach().cpu().tolist())
            imp_mse_all.extend(imp_mse.detach().cpu().tolist())
            imp_mae_all.extend(imp_mae.detach().cpu().tolist())
            occ_bce_all.extend(occ_bce.detach().cpu().tolist())
            occ_match_all.extend(occ_match.detach().cpu().tolist())
            occ_exact_all.extend(occ_exact.detach().cpu().tolist())
            kl_total_all.extend(kl_total.detach().cpu().tolist())

            hm_only_occ_bce.extend(hm_only_occ_bce_batch.detach().cpu().tolist())
            hm_only_imp_mse.extend(hm_only_imp_mse_batch.detach().cpu().tolist())
            imp_only_occ_bce.extend(imp_only_occ_bce_batch.detach().cpu().tolist())
            imp_only_hm_mse.extend(imp_only_hm_mse_batch.detach().cpu().tolist())

            # Latent aggregates
            mu_sum += mu.sum(dim=0)
            mu_sumsq += (mu**2).sum(dim=0)
            sigma_sum += torch.exp(0.5 * logvar).sum(dim=0)
            kl_dim_sum += kl_per_dim.sum(dim=0)
            latent_n += int(mu.shape[0])

            # Real-data simple stats (for later real-vs-gen comparison)
            real_occ_sum += (occ > 0.5).to(torch.float32).sum(dim=0)
            real_hm_sum += float(hm.sum().item())
            real_hm_sumsq += float((hm**2).sum().item())
            real_imp_sum += float(imp.sum().item())
            real_imp_sumsq += float((imp**2).sum().item())
            real_hm_n += int(hm.numel())
            real_imp_n += int(imp.numel())

    if latent_n <= 0:
        raise SystemExit("No latent samples accumulated")
    mu_mean = (mu_sum / float(latent_n)).detach().cpu().numpy()
    mu_var = (mu_sumsq / float(latent_n)).detach().cpu().numpy() - mu_mean**2
    sigma_mean = (sigma_sum / float(latent_n)).detach().cpu().numpy()
    kl_per_dim_mean = (kl_dim_sum / float(latent_n)).detach().cpu().numpy()

    active_units = int(np.sum(mu_var > float(cfg.active_var_thresh)))

    # Real-vs-gen stats
    print(f"Generating {int(cfg.n_gen)} samples for real-vs-generated stats...")
    k_eval_np = np.asarray(k_all, dtype=np.int64)
    if k_eval_np.size == 0:
        raise SystemExit("No evaluated samples; cannot sample K distribution")

    n_gen = int(cfg.n_gen)
    k_gen = rng.choice(k_eval_np, size=n_gen, replace=True)
    k_gen_t = torch.from_numpy(k_gen).to(device=device, dtype=torch.long)

    with torch.no_grad():
        # NOTE: model.inference() only uses per_K_latent_stats when K is a single int.
        # Here K varies per sample, so we sample z manually using per-K tables when available.
        global_mu, global_std = _extract_latent_mu_std(latent_stats=loaded.latent_stats, latent_dim=cfg_latent_dim)
        mu_table, std_table = _build_per_k_latent_tables(
            per_k_latent_stats=loaded.per_k_latent_stats,
            global_mu=global_mu,
            global_std=global_std,
            latent_dim=cfg_latent_dim,
        )

        mu_gen = torch.from_numpy(mu_table[k_gen]).to(device=device)
        std_gen = torch.from_numpy(std_table[k_gen]).to(device=device)
        z_gen = torch.randn((n_gen, cfg_latent_dim), device=device) * (std_gen * float(cfg.shared_temp)) + mu_gen

        hm_gen, occ_logits_gen, imp_gen = model.decode(z_gen, k_gen_t)
        occ_prob_gen = torch.sigmoid(occ_logits_gen)

    # Convert generated occupancy probs to exactly-K binary vectors (as in inference)
    occ_gen_bin = torch.zeros_like(occ_prob_gen)
    for i in range(n_gen):
        kk = int(k_gen[i])
        if kk <= 0:
            continue
        idx = torch.topk(occ_prob_gen[i], min(kk, occ_prob_gen.shape[-1])).indices
        occ_gen_bin[i, idx] = 1.0

    real_occ_rate = (real_occ_sum / float(len(eval_ids))).detach().cpu().numpy()
    gen_occ_rate = occ_gen_bin.mean(dim=0).detach().cpu().numpy()
    occ_rate_l1 = float(np.mean(np.abs(real_occ_rate - gen_occ_rate)))

    # Real/gen modality mean/std in *normalized* space
    real_hm_mean = real_hm_sum / float(real_hm_n)
    real_hm_std = float(np.sqrt(max(real_hm_sumsq / float(real_hm_n) - real_hm_mean**2, 0.0)))

    real_imp_mean = real_imp_sum / float(real_imp_n)
    real_imp_std = float(np.sqrt(max(real_imp_sumsq / float(real_imp_n) - real_imp_mean**2, 0.0)))

    hm_gen_mean = float(hm_gen.mean().item())
    hm_gen_std = float(hm_gen.std(unbiased=False).item())
    imp_gen_mean = float(imp_gen.mean().item())
    imp_gen_std = float(imp_gen.std(unbiased=False).item())

    # Aggregate recon/latent metrics
    k_arr = np.asarray(k_all, dtype=np.int64)
    hm_mse_arr = np.asarray(hm_mse_all, dtype=np.float64)
    hm_mae_arr = np.asarray(hm_mae_all, dtype=np.float64)
    imp_mse_arr = np.asarray(imp_mse_all, dtype=np.float64)
    imp_mae_arr = np.asarray(imp_mae_all, dtype=np.float64)
    occ_bce_arr = np.asarray(occ_bce_all, dtype=np.float64)
    occ_match_arr = np.asarray(occ_match_all, dtype=np.float64)
    occ_exact_arr = np.asarray(occ_exact_all, dtype=np.float64)
    kl_total_arr = np.asarray(kl_total_all, dtype=np.float64)

    metrics = {
        "checkpoint": str(cfg.checkpoint),
        "epoch": loaded.epoch,
        "device": str(device),
        "n_eval": int(len(eval_ids)),
        "n_gen": int(n_gen),
        "shared_temp": float(cfg.shared_temp),
        "gen_sampling": {
            "uses_per_k_tables": bool(loaded.per_k_latent_stats is not None),
        },
        "recon_full": {
            "heatmap_mse_mean": float(np.mean(hm_mse_arr)),
            "heatmap_mae_mean": float(np.mean(hm_mae_arr)),
            "impedance_mse_mean": float(np.mean(imp_mse_arr)),
            "impedance_mae_mean": float(np.mean(imp_mae_arr)),
            "occupancy_bce_mean": float(np.mean(occ_bce_arr)),
            "occupancy_topk_match_mean": float(np.mean(occ_match_arr)),
            "occupancy_exact_match_mean": float(np.mean(occ_exact_arr)),
        },
        "recon_cross_modal": {
            "heatmap_only": {
                "occupancy_bce_mean": float(np.mean(np.asarray(hm_only_occ_bce, dtype=np.float64))),
                "impedance_mse_mean": float(np.mean(np.asarray(hm_only_imp_mse, dtype=np.float64))),
            },
            "impedance_only": {
                "occupancy_bce_mean": float(np.mean(np.asarray(imp_only_occ_bce, dtype=np.float64))),
                "heatmap_mse_mean": float(np.mean(np.asarray(imp_only_hm_mse, dtype=np.float64))),
            },
        },
        "latent": {
            "latent_dim": int(cfg_latent_dim),
            "kl_total_mean": float(np.mean(kl_total_arr)),
            "kl_per_dim_mean": [float(x) for x in kl_per_dim_mean.tolist()],
            "active_units": int(active_units),
            "active_var_thresh": float(cfg.active_var_thresh),
            "mu_mean_per_dim": [float(x) for x in mu_mean.tolist()],
            "mu_var_per_dim": [float(x) for x in mu_var.tolist()],
            "sigma_mean_per_dim": [float(x) for x in sigma_mean.tolist()],
        },
        "real_vs_gen": {
            "occ_rate_l1_mean": occ_rate_l1,
            "heatmap_mean_real": float(real_hm_mean),
            "heatmap_std_real": float(real_hm_std),
            "heatmap_mean_gen": hm_gen_mean,
            "heatmap_std_gen": hm_gen_std,
            "imp_mean_real": float(real_imp_mean),
            "imp_std_real": float(real_imp_std),
            "imp_mean_gen": imp_gen_mean,
            "imp_std_gen": imp_gen_std,
        },
    }

    # Latent analysis (inspired by experiments/exp027_sigma_reg_tuning/codes/visualize_latent.py)
    mu_mat = np.concatenate(mu_batches, axis=0).astype(np.float64, copy=False)
    if mu_mat.shape[0] != k_arr.shape[0]:
        raise SystemExit(f"Internal error: mu rows {mu_mat.shape[0]} != K rows {k_arr.shape[0]}")

    sigma_mean_per_sample_arr = np.asarray(fused_sigma_mean_per_sample, dtype=np.float64)
    if sigma_mean_per_sample_arr.shape[0] != k_arr.shape[0]:
        raise SystemExit(
            f"Internal error: sigma rows {sigma_mean_per_sample_arr.shape[0]} != K rows {k_arr.shape[0]}"
        )

    pca_mean, pca_components, _pca_ev, pca_evr = _pca_from_mu(mu_mat)
    cumev = np.cumsum(pca_evr)
    dims_90 = int(min(int(np.searchsorted(cumev, 0.90)) + 1, int(cumev.size))) if cumev.size else 0
    dims_95 = int(min(int(np.searchsorted(cumev, 0.95)) + 1, int(cumev.size))) if cumev.size else 0
    dims_99 = int(min(int(np.searchsorted(cumev, 0.99)) + 1, int(cumev.size))) if cumev.size else 0

    mu_std_per_dim = np.sqrt(np.maximum(mu_var, 0.0))

    r2_k_probe, mae_k_probe, k_true_probe, k_pred_probe = _linear_probe_k_from_mu(mu_mat, k_arr, seed=int(cfg.seed))

    counts_f = k_counts.astype(np.float64)
    expert_by_k: dict[str, dict[str, np.ndarray]] = {}
    if float(np.sum(counts_f)) > 0.0:
        for m in ("heatmap", "occupancy", "impedance"):
            diff_sum = expert_acc[m]["mu_rms_diff_sum"]
            diff_sumsq = expert_acc[m]["mu_rms_diff_sumsq"]
            sig_sum = expert_acc[m]["sigma_sum"]
            sig_sumsq = expert_acc[m]["sigma_sumsq"]

            diff_mean = np.divide(diff_sum, counts_f, out=np.full_like(diff_sum, np.nan), where=counts_f > 0)
            diff_var = (
                np.divide(diff_sumsq, counts_f, out=np.full_like(diff_sumsq, np.nan), where=counts_f > 0)
                - diff_mean**2
            )
            diff_std = np.sqrt(np.maximum(diff_var, 0.0))

            sig_mean = np.divide(sig_sum, counts_f, out=np.full_like(sig_sum, np.nan), where=counts_f > 0)
            sig_var = (
                np.divide(sig_sumsq, counts_f, out=np.full_like(sig_sumsq, np.nan), where=counts_f > 0)
                - sig_mean**2
            )
            sig_std = np.sqrt(np.maximum(sig_var, 0.0))

            expert_by_k[m] = {
                "mu_rms_diff_mean": diff_mean,
                "mu_rms_diff_std": diff_std,
                "sigma_mean": sig_mean,
                "sigma_std": sig_std,
            }

    # Write per-sample CSV
    per_sample_csv = out_dir / "vae_eval_per_sample.csv"
    with open(per_sample_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "sample_id",
                "K",
                "hm_mse",
                "hm_mae",
                "imp_mse",
                "imp_mae",
                "occ_bce",
                "occ_topk_match",
                "occ_exact_match",
                "kl_total",
            ]
        )
        for sid, kk, a, b, c, d, e, g, ex, kl in zip(
            eval_ids,
            k_arr.tolist(),
            hm_mse_arr.tolist(),
            hm_mae_arr.tolist(),
            imp_mse_arr.tolist(),
            imp_mae_arr.tolist(),
            occ_bce_arr.tolist(),
            occ_match_arr.tolist(),
            occ_exact_arr.tolist(),
            kl_total_arr.tolist(),
        ):
            w.writerow(
                [
                    int(sid),
                    int(kk),
                    float(a),
                    float(b),
                    float(c),
                    float(d),
                    float(e),
                    float(g),
                    float(ex),
                    float(kl),
                ]
            )

    # Plots
    recon_plot = plots_dir / "recon_error_vs_k.png"
    kl_plot = plots_dir / "kl_per_dim.png"
    occ_plot = plots_dir / "occ_rate_real_vs_gen.png"

    _plot_recon_vs_k(k=k_arr, hm_mse=hm_mse_arr, imp_mse=imp_mse_arr, occ_bce=occ_bce_arr, out_path=recon_plot)
    _plot_kl_per_dim(kl_per_dim=kl_per_dim_mean, out_path=kl_plot)
    _plot_occ_rates(occ_real_rate=real_occ_rate, occ_gen_rate=gen_occ_rate, out_path=occ_plot)

    # Latent analysis plots
    latent_pca_scatter_plot = plots_dir / "latent_pca_scatter.png"
    latent_pca_variance_plot = plots_dir / "latent_pca_variance.png"
    latent_per_dim_plot = plots_dir / "latent_per_dim_mu_sigma.png"
    latent_sigma_by_k_plot = plots_dir / "latent_sigma_by_k_bucket.png"
    latent_kl_by_k_plot = plots_dir / "latent_kl_by_k_bucket.png"
    latent_probe_plot = plots_dir / "latent_probe_K_from_mu.png"
    latent_expert_plot = plots_dir / "latent_expert_vs_fused_by_K.png"

    _plot_latent_pca_scatter(
        mu=mu_mat,
        K=k_arr,
        mean=pca_mean,
        components=pca_components,
        evr=pca_evr,
        out_path=latent_pca_scatter_plot,
        seed=int(cfg.seed),
    )
    _plot_latent_pca_variance(
        evr=pca_evr,
        dims_90=dims_90,
        dims_95=dims_95,
        dims_99=dims_99,
        out_path=latent_pca_variance_plot,
    )

    dead_sigma_thresh = 0.95
    dead_mu_std_thresh = 0.05
    dead_dims = _plot_latent_per_dim_mu_sigma(
        mu_std=mu_std_per_dim,
        sigma_mean=sigma_mean,
        out_path=latent_per_dim_plot,
        dead_sigma_thresh=dead_sigma_thresh,
        dead_mu_std_thresh=dead_mu_std_thresh,
    )

    sigma_bucket_stats = _plot_hist_by_k_bucket(
        values=sigma_mean_per_sample_arr,
        K=k_arr,
        out_path=latent_sigma_by_k_plot,
        xlabel="Mean fused σ per sample",
        title="Fused σ distribution by K bucket",
        target_vlines=[(0.45, "--", "orange", "σ≈0.45"), (1.0, "--", "grey", "prior σ=1")],
    )
    kl_bucket_stats = _plot_hist_by_k_bucket(
        values=kl_total_arr,
        K=k_arr,
        out_path=latent_kl_by_k_plot,
        xlabel="Total KL per sample (nats)",
        title="Total KL distribution by K bucket",
        target_vlines=None,
    )

    _plot_probe_k_from_mu(
        y_true=k_true_probe,
        y_pred=k_pred_probe,
        r2=float(r2_k_probe),
        mae=float(mae_k_probe),
        out_path=latent_probe_plot,
    )
    if expert_by_k:
        _plot_expert_disagreement_by_k(K_counts=k_counts, per_modality=expert_by_k, out_path=latent_expert_plot)

    metrics["latent_analysis"] = {
        "pca": {
            "dims_90": int(dims_90),
            "dims_95": int(dims_95),
            "dims_99": int(dims_99),
            "explained_variance_ratio": [float(x) for x in np.asarray(pca_evr, dtype=np.float64).tolist()],
        },
        "per_dim": {
            "mu_std_per_dim": [float(x) for x in np.asarray(mu_std_per_dim, dtype=np.float64).tolist()],
            "dead_dims": int(dead_dims),
            "dead_sigma_thresh": float(dead_sigma_thresh),
            "dead_mu_std_thresh": float(dead_mu_std_thresh),
        },
        "fused_sigma": {
            "mean_sigma_per_sample_mean": float(np.mean(sigma_mean_per_sample_arr)),
            "mean_sigma_per_sample_std": float(np.std(sigma_mean_per_sample_arr)),
            "by_k_bucket": sigma_bucket_stats,
        },
        "kl_total_by_k_bucket": kl_bucket_stats,
        "probe_K_from_mu": {
            "r2": float(r2_k_probe),
            "mae": float(mae_k_probe),
        },
        "expert_vs_fused_by_K": {
            "K_counts": [int(x) for x in k_counts.tolist()],
            "heatmap": {
                "mu_rms_diff_mean": _nan_to_none_list(expert_by_k.get("heatmap", {}).get("mu_rms_diff_mean", np.full((53,), np.nan))),
                "mu_rms_diff_std": _nan_to_none_list(expert_by_k.get("heatmap", {}).get("mu_rms_diff_std", np.full((53,), np.nan))),
                "sigma_mean": _nan_to_none_list(expert_by_k.get("heatmap", {}).get("sigma_mean", np.full((53,), np.nan))),
                "sigma_std": _nan_to_none_list(expert_by_k.get("heatmap", {}).get("sigma_std", np.full((53,), np.nan))),
            },
            "occupancy": {
                "mu_rms_diff_mean": _nan_to_none_list(expert_by_k.get("occupancy", {}).get("mu_rms_diff_mean", np.full((53,), np.nan))),
                "mu_rms_diff_std": _nan_to_none_list(expert_by_k.get("occupancy", {}).get("mu_rms_diff_std", np.full((53,), np.nan))),
                "sigma_mean": _nan_to_none_list(expert_by_k.get("occupancy", {}).get("sigma_mean", np.full((53,), np.nan))),
                "sigma_std": _nan_to_none_list(expert_by_k.get("occupancy", {}).get("sigma_std", np.full((53,), np.nan))),
            },
            "impedance": {
                "mu_rms_diff_mean": _nan_to_none_list(expert_by_k.get("impedance", {}).get("mu_rms_diff_mean", np.full((53,), np.nan))),
                "mu_rms_diff_std": _nan_to_none_list(expert_by_k.get("impedance", {}).get("mu_rms_diff_std", np.full((53,), np.nan))),
                "sigma_mean": _nan_to_none_list(expert_by_k.get("impedance", {}).get("sigma_mean", np.full((53,), np.nan))),
                "sigma_std": _nan_to_none_list(expert_by_k.get("impedance", {}).get("sigma_std", np.full((53,), np.nan))),
            },
        },
    }

    # Report
    report_path = out_dir / "vae_eval_report.md"

    lines: list[str] = []
    lines.append("# VAE evaluation report\n")
    lines.append(f"Checkpoint: `{cfg.checkpoint}`  ")
    if loaded.epoch is not None:
        lines.append(f"Epoch: `{loaded.epoch}`  ")
    lines.append(f"Device: `{device}`  ")
    lines.append(f"Eval samples: `{len(eval_ids)}`  Generated samples: `{n_gen}`\n")

    lines.append("## Headline metrics (full multi-modal recon)\n")
    lines.append(f"- Heatmap MSE (z-score): {_fmt(metrics['recon_full']['heatmap_mse_mean'])}")
    lines.append(f"- Heatmap MAE (z-score): {_fmt(metrics['recon_full']['heatmap_mae_mean'])}")
    lines.append(f"- Impedance MSE (norm): {_fmt(metrics['recon_full']['impedance_mse_mean'])}")
    lines.append(f"- Impedance MAE (norm): {_fmt(metrics['recon_full']['impedance_mae_mean'])}")
    lines.append(f"- Occupancy BCE (logits): {_fmt(metrics['recon_full']['occupancy_bce_mean'])}")
    lines.append(f"- Occupancy top-K match: {_fmt(metrics['recon_full']['occupancy_topk_match_mean'])}")
    lines.append(f"- Occupancy exact match: {_fmt(metrics['recon_full']['occupancy_exact_match_mean'])}\n")

    lines.append("## Cross-modal recon (single-modality encoder)\n")
    lines.append(
        "- Heatmap-only → occupancy BCE: "
        + _fmt(metrics["recon_cross_modal"]["heatmap_only"]["occupancy_bce_mean"])
        + "; impedance MSE: "
        + _fmt(metrics["recon_cross_modal"]["heatmap_only"]["impedance_mse_mean"])
    )
    lines.append(
        "- Impedance-only → occupancy BCE: "
        + _fmt(metrics["recon_cross_modal"]["impedance_only"]["occupancy_bce_mean"])
        + "; heatmap MSE: "
        + _fmt(metrics["recon_cross_modal"]["impedance_only"]["heatmap_mse_mean"])
        + "\n"
    )

    lines.append("## Latent diagnostics\n")
    lines.append(f"- Latent dim: `{cfg_latent_dim}`")
    lines.append(f"- Mean KL (total): {_fmt(metrics['latent']['kl_total_mean'])}")
    lines.append(f"- Active units (var(mu) > {float(cfg.active_var_thresh):.0e}): `{active_units}`\n")

    lines.append("## Latent analysis (held-out encodings)\n")
    lines.append(f"- PCA dims for 90/95/99% variance: `{dims_90}` / `{dims_95}` / `{dims_99}`")
    lines.append(
        "- Dead dims (σ>"
        + _fmt(dead_sigma_thresh)
        + " & μ-std<"
        + _fmt(dead_mu_std_thresh)
        + "): `"
        + str(int(dead_dims))
        + " / "
        + str(int(cfg_latent_dim))
        + "`"
    )
    lines.append(
        "- Linear probe K←μ (80/20 split): R² "
        + _fmt(float(r2_k_probe))
        + "; MAE "
        + _fmt(float(mae_k_probe))
        + "\n"
    )

    lines.append("## Real vs generated (basic moments, normalized space)\n")
    lines.append(f"- Occupancy rate mean |real-gen| (52 slots): {_fmt(metrics['real_vs_gen']['occ_rate_l1_mean'])}")
    lines.append(
        "- Heatmap mean/std (real → gen): "
        + _fmt(metrics["real_vs_gen"]["heatmap_mean_real"])
        + "/"
        + _fmt(metrics["real_vs_gen"]["heatmap_std_real"])
        + " → "
        + _fmt(metrics["real_vs_gen"]["heatmap_mean_gen"])
        + "/"
        + _fmt(metrics["real_vs_gen"]["heatmap_std_gen"])
    )
    lines.append(
        "- Impedance mean/std (real → gen): "
        + _fmt(metrics["real_vs_gen"]["imp_mean_real"])
        + "/"
        + _fmt(metrics["real_vs_gen"]["imp_std_real"])
        + " → "
        + _fmt(metrics["real_vs_gen"]["imp_mean_gen"])
        + "/"
        + _fmt(metrics["real_vs_gen"]["imp_std_gen"])
        + "\n"
    )

    lines.append("## Plots\n")
    lines.append(f"![Recon error vs K](plots/{recon_plot.name})\n")
    lines.append(f"![KL per dim](plots/{kl_plot.name})\n")
    lines.append(f"![Occupancy rates](plots/{occ_plot.name})\n")

    lines.append(f"![Latent PCA scatter](plots/{latent_pca_scatter_plot.name})\n")
    lines.append(f"![Latent PCA variance](plots/{latent_pca_variance_plot.name})\n")
    lines.append(f"![Latent per-dim μ/σ](plots/{latent_per_dim_plot.name})\n")
    lines.append(f"![Fused σ by K bucket](plots/{latent_sigma_by_k_plot.name})\n")
    lines.append(f"![Total KL by K bucket](plots/{latent_kl_by_k_plot.name})\n")
    lines.append(f"![Probe: K from fused μ](plots/{latent_probe_plot.name})\n")
    if expert_by_k:
        lines.append(f"![Experts vs fused (by K)](plots/{latent_expert_plot.name})\n")

    lines.append("## Files\n")
    lines.append(f"- Per-sample CSV: `{per_sample_csv}`")
    lines.append(f"- Metrics JSON: `{out_dir / 'vae_eval_metrics.json'}`")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    metrics_path = out_dir / "vae_eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n✓ Wrote evaluation outputs")
    print(f"  Report:  {report_path}")
    print(f"  Plots:   {plots_dir}")
    print(f"  CSV:     {per_sample_csv}")
    print(f"  Metrics: {metrics_path}")


def main() -> None:
    run_eval(CONFIG)


if __name__ == "__main__":
    main()
