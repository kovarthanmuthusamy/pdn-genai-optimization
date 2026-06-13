"""
visualize_latent.py — Latent space visualizations for exp042 (PI_freq-conditioned VAE).

Encodes a multifreq dataset subset and writes:
  - t-SNE / PCA colored by K and by PI frequency (MHz)
  - PCA variance, per-dim mu/sigma, KL diagnostics
  - Decoder grids along top-2 PCs at several PI_freq (MHz)
  - Linear probes: K←mu and MHz←mu
  - Expert-vs-fused disagreement vs K and vs MHz

Usage:
    cd ~/gan
    python experiments/exp042/codes/visualize_latent.py
    python experiments/exp042/codes/visualize_latent.py --ckpt checkpoints/last_model.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

_CODES = Path(__file__).resolve().parent
if str(_CODES) not in sys.path:
    sys.path.insert(0, str(_CODES))

from exp043_eval_common import (  # noqa: E402
    heatmap_to_physical,
    ANCHOR_MHZ,
    encode_dataset,
    load_exp_config,
    load_model,
    pi_tensor_mhz,
    resolve_paths,
)

TSNE_PERPLEXITY = 40
MAX_SAMPLES_DEFAULT = 30_000
GRID_MHZ = (63.0, 200.0, 400.0)
GRID_K = 26


def _k_cmap(K_arr):
    return Normalize(vmin=0, vmax=52)


def _mhz_cmap(mhz_arr):
    lo, hi = float(np.nanmin(mhz_arr)), float(np.nanmax(mhz_arr))
    if hi <= lo:
        hi = lo + 1.0
    return Normalize(vmin=lo, vmax=hi)


def kl_diag_standard_normal(mu, logvar):
    var = np.exp(logvar)
    return 0.5 * (mu ** 2 + var - 1.0 - logvar)


def plot_tsne(mu, color_arr, norm, out_path: Path, title: str, cbar_label: str):
    print(f"\nComputing t-SNE for {out_path.name} …")
    emb = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, random_state=42, max_iter=1000).fit_transform(mu)
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=color_arr, cmap="plasma", norm=norm, s=8, alpha=0.65, linewidths=0)
    plt.colorbar(sc, ax=ax, label=cbar_label)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path.name}")


def plot_pca_scatter(mu, color_arr, norm, out_path: Path, title: str, cbar_label: str):
    pca = PCA(n_components=10, random_state=42)
    pca.fit(mu)
    emb2 = pca.transform(mu)[:, :2]
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(emb2[:, 0], emb2[:, 1], c=color_arr, cmap="plasma", norm=norm, s=8, alpha=0.65, linewidths=0)
    plt.colorbar(sc, ax=ax, label=cbar_label)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path.name}")
    return pca


def plot_pca_variance(mu, out_dir: Path):
    pca_full = PCA(n_components=min(mu.shape[1], mu.shape[0]), random_state=42)
    pca_full.fit(mu)
    ev, cumev = pca_full.explained_variance_ratio_, np.cumsum(pca_full.explained_variance_ratio_)
    dims_90 = int(np.searchsorted(cumev, 0.90)) + 1
    dims_95 = int(np.searchsorted(cumev, 0.95)) + 1
    print(f"  PCA dims for 90%/95% variance: {dims_90} / {dims_95} of {mu.shape[1]}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.bar(range(1, min(31, len(ev) + 1)), ev[:30] * 100, color="steelblue", alpha=0.8)
    ax1.set_xlabel("PC")
    ax1.set_ylabel("Explained variance (%)")
    ax1.set_title("Per-component variance (top 30)")
    ax1.grid(True, axis="y", alpha=0.35)

    ax2.plot(range(1, len(cumev) + 1), cumev * 100, color="steelblue", lw=2)
    for thr, d, col in [(90, dims_90, "gold"), (95, dims_95, "orange")]:
        ax2.axhline(thr, ls="--", color=col, alpha=0.7)
        ax2.axvline(d, ls="--", color=col, alpha=0.7, label=f"{thr}% @ dim {d}")
    ax2.set_xlabel("Number of components")
    ax2.set_ylabel("Cumulative variance (%)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_dir / "plot3_pca_variance.png", dpi=150)
    plt.close(fig)
    print("  → plot3_pca_variance.png")
    return pca_full


def plot_per_dim(mu, sigma, out_dir: Path):
    dim_mu_std = mu.std(axis=0)
    dim_sigma_mean = sigma.mean(axis=0)
    order = np.argsort(dim_mu_std)[::-1]
    dead = int(((dim_sigma_mean > 0.95) & (dim_mu_std < 0.05)).sum())
    print(f"  Dead dims: {dead}/{mu.shape[1]}  mean σ={dim_sigma_mean.mean():.4f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    x = np.arange(mu.shape[1])
    ax1.bar(x, dim_mu_std[order], color="#4C72B0", alpha=0.8)
    ax1.set_ylabel("μ std")
    ax1.set_title("Per-dim posterior spread (sorted)")
    ax2.bar(x, dim_sigma_mean[order], color="#55A868", alpha=0.8)
    ax2.axhline(0.45, ls="--", color="orange", label="target σ=0.45")
    ax2.set_ylabel("Mean posterior σ")
    ax2.set_xlabel("Latent dim (sorted)")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "plot4_per_dim_mu_sigma.png", dpi=150)
    plt.close(fig)
    print("  → plot4_per_dim_mu_sigma.png")


def plot_decoder_grid(model, pca, device, paths, mhz: float, K_val: int, grid_n: int, out_dir: Path):
    norm_stats = json.load(open(paths["norm_stats"], encoding="utf-8"))
    binary_mask = np.load(paths["binary_mask"])

    pc1 = torch.tensor(pca.components_[0], dtype=torch.float32, device=device)
    pc2 = torch.tensor(pca.components_[1], dtype=torch.float32, device=device)
    origin = torch.tensor(pca.mean_, dtype=torch.float32, device=device)
    pc1_std = float(np.sqrt(pca.explained_variance_[0]))
    pc2_std = float(np.sqrt(pca.explained_variance_[1]))
    rng1 = np.linspace(-3 * pc1_std, 3 * pc1_std, grid_n)
    rng2 = np.linspace(-3 * pc2_std, 3 * pc2_std, grid_n)[::-1]

    K_tensor = torch.full((1,), K_val, dtype=torch.long, device=device)
    pi = pi_tensor_mhz(mhz, 1, device)

    fig, axes = plt.subplots(grid_n, grid_n, figsize=(grid_n * 2.2, grid_n * 2.2))
    fig.suptitle(f"Decoder grid PC1×PC2  K={K_val}  PI={mhz:.0f} MHz", fontsize=13, fontweight="bold")
    with torch.no_grad():
        for row, a2 in enumerate(rng2):
            for col, a1 in enumerate(rng1):
                z = (origin + a1 * pc1 + a2 * pc2).unsqueeze(0)
                hm, _, _ = model.decode(z, K_tensor, pi)
                hm_phys = heatmap_to_physical(hm, norm_stats)
                hm_np = np.ma.masked_where(~binary_mask, hm_phys[0, 0].cpu().numpy())
                ax = axes[row][col]
                ax.imshow(hm_np, cmap="jet", aspect="auto", origin="lower", interpolation="bicubic")
                ax.axis("off")
    fig.savefig(out_dir / f"plot5_decoder_grid_K{K_val}_f{int(mhz)}MHz.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → plot5_decoder_grid_K{K_val}_f{int(mhz)}MHz.png")


def plot_sigma_by_buckets(sigma, labels, bucket_fn, out_path: Path, title: str):
    sigma_mean = sigma.mean(axis=1)
    buckets = bucket_fn()
    fig, axes = plt.subplots(1, len(buckets), figsize=(3.5 * len(buckets), 4), sharey=True)
    if len(buckets) == 1:
        axes = [axes]
    for ax, (mask, label, col) in zip(axes, buckets):
        vals = sigma_mean[mask]
        if len(vals) == 0:
            ax.set_title(f"{label}\n(empty)")
            continue
        ax.hist(vals, bins=25, color=col, alpha=0.85, density=True)
        ax.axvline(vals.mean(), color="black", lw=1.5, label=f"μ={vals.mean():.3f}")
        ax.axvline(0.45, color="orange", ls="--", label="target")
        ax.set_title(f"{label} n={mask.sum()}")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Density")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path.name}")


def _k_buckets(K_arr):
    specs = [(0, 13, "K 0–12"), (13, 26, "K 13–25"), (26, 39, "K 26–38"), (39, 53, "K 39–52")]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    return [((K_arr >= lo) & (K_arr < hi), lbl, c) for (lo, hi, lbl), c in zip(specs, colors)]


def _mhz_anchor_buckets(mhz_arr):
    out = []
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(ANCHOR_MHZ)))
    for ai, anchor in enumerate(ANCHOR_MHZ):
        if ai == 0:
            mask = mhz_arr <= (ANCHOR_MHZ[0] + ANCHOR_MHZ[1]) / 2 if len(ANCHOR_MHZ) > 1 else np.ones_like(mhz_arr, dtype=bool)
        elif ai == len(ANCHOR_MHZ) - 1:
            mask = mhz_arr > (ANCHOR_MHZ[-2] + ANCHOR_MHZ[-1]) / 2
        else:
            lo = (ANCHOR_MHZ[ai - 1] + anchor) / 2
            hi = (anchor + ANCHOR_MHZ[ai + 1]) / 2
            mask = (mhz_arr > lo) & (mhz_arr <= hi)
        out.append((mask, f"{anchor:.0f} MHz", colors[ai]))
    return out


def plot_probe(y, pred_labels, out_path: Path, title: str, xlabel: str):
    r2 = float(r2_score(y, pred_labels))
    mae = float(np.mean(np.abs(y - pred_labels)))
    print(f"  {title}: R²={r2:.4f}  MAE={mae:.3f}")
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y, pred_labels, s=10, alpha=0.35, linewidths=0)
    lo, hi = float(min(y.min(), pred_labels.min())), float(max(y.max(), pred_labels.max()))
    ax.plot([lo, hi], [lo, hi], "k-", lw=1)
    ax.set_title(f"{title}\nR²={r2:.3f}  MAE={mae:.2f}")
    ax.set_xlabel(f"True {xlabel}")
    ax.set_ylabel(f"Predicted {xlabel}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path.name}")


def plot_probes(mu, K_arr, mhz_arr, out_dir: Path):
    X_tr, X_te, yK_tr, yK_te = train_test_split(mu, K_arr.astype(np.float32), test_size=0.2, random_state=42)
    reg_k = LinearRegression().fit(X_tr, yK_tr)
    plot_probe(yK_te, reg_k.predict(X_te), out_dir / "plot9_probe_K_from_mu.png",
               "Linear probe: K from fused μ", "K")

    X_tr, X_te, yF_tr, yF_te = train_test_split(mu, mhz_arr.astype(np.float32), test_size=0.2, random_state=42)
    reg_f = LinearRegression().fit(X_tr, yF_tr)
    plot_probe(yF_te, reg_f.predict(X_te), out_dir / "plot9b_probe_MHz_from_mu.png",
               "Linear probe: PI_freq (MHz) from fused μ", "MHz")


def plot_kl_by_buckets(mu, logvar, bucket_fn, out_path: Path, title: str):
    kl_total = kl_diag_standard_normal(mu, logvar).sum(axis=1)
    buckets = bucket_fn()
    fig, axes = plt.subplots(1, len(buckets), figsize=(3.5 * len(buckets), 4), sharey=True)
    if len(buckets) == 1:
        axes = [axes]
    for ax, (mask, label, col) in zip(axes, buckets):
        vals = kl_total[mask]
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=30, color=col, alpha=0.85, density=True)
        ax.axvline(vals.mean(), color="black", lw=1.5)
        ax.set_title(f"{label} n={mask.sum()}")
    axes[0].set_ylabel("Density")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path.name}")


def main():
    ap = argparse.ArgumentParser(description="exp042 latent visualization (PI_freq)")
    ap.add_argument("--ckpt", default=None, help="Checkpoint .pt (default: config / last_model)")
    ap.add_argument("--max-samples", type=int, default=MAX_SAMPLES_DEFAULT)
    ap.add_argument("--output", default=None, help="Output directory")
    ap.add_argument("--grid-mhz", nargs="*", type=float, default=list(GRID_MHZ))
    args = ap.parse_args()

    cfg = load_exp_config()
    paths = resolve_paths(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.ckpt) if args.ckpt else Path(paths["checkpoint"])
    out_dir = Path(args.output or (paths["exp_dir"] / "latent_visuals"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}\nCheckpoint: {ckpt}\nData: {paths['data_dir']}\nOutput: {out_dir}\n")

    model, _, _, _ = load_model(ckpt, device)
    enc = encode_dataset(model, paths["data_dir"], device, max_samples=args.max_samples)
    mu, logvar, sigma = enc["mu"], enc["logvar"], enc["sigma"]
    K_arr, mhz_arr = enc["K"], enc["mhz"]

    plot_tsne(mu, K_arr, _k_cmap(K_arr), out_dir / "plot1_tsne_by_K.png",
              "t-SNE of fused μ (colored by K)", "K")
    plot_tsne(mu, mhz_arr, _mhz_cmap(mhz_arr), out_dir / "plot1b_tsne_by_MHz.png",
              "t-SNE of fused μ (colored by PI_freq MHz)", "MHz")

    plot_pca_scatter(mu, K_arr, _k_cmap(K_arr), out_dir / "plot2_pca_scatter_by_K.png",
                     "PCA of fused μ (colored by K)", "K")
    pca_full = plot_pca_variance(mu, out_dir)
    plot_pca_scatter(mu, mhz_arr, _mhz_cmap(mhz_arr), out_dir / "plot2b_pca_scatter_by_MHz.png",
                     "PCA of fused μ (colored by MHz)", "MHz")

    plot_per_dim(mu, sigma, out_dir)
    for mhz in args.grid_mhz:
        plot_decoder_grid(model, pca_full, device, paths, mhz, GRID_K, 8, out_dir)

    plot_sigma_by_buckets(sigma, K_arr, lambda: _k_buckets(K_arr),
                          out_dir / "plot6_sigma_by_K.png", "Fused σ by K bucket")
    plot_sigma_by_buckets(sigma, mhz_arr, lambda: _mhz_anchor_buckets(mhz_arr),
                          out_dir / "plot6b_sigma_by_MHz_anchor.png", "Fused σ by PI anchor bucket")

    kl_dim = kl_diag_standard_normal(mu, logvar).mean(axis=0)
    order = np.argsort(kl_dim)[::-1]
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.bar(np.arange(len(kl_dim)), kl_dim[order], color="#4C72B0", alpha=0.85)
    ax.set_title("KL per latent dim (sorted)")
    ax.set_xlabel("Dim")
    ax.set_ylabel("Mean KL (nats)")
    fig.tight_layout()
    fig.savefig(out_dir / "plot7_kl_per_dim.png", dpi=150)
    plt.close(fig)
    print("  → plot7_kl_per_dim.png")

    plot_kl_by_buckets(mu, logvar, lambda: _k_buckets(K_arr),
                       out_dir / "plot8_kl_by_K.png", "Total KL by K bucket")
    plot_kl_by_buckets(mu, logvar, lambda: _mhz_anchor_buckets(mhz_arr),
                       out_dir / "plot8b_kl_by_MHz_anchor.png", "Total KL by MHz anchor bucket")

    plot_probes(mu, K_arr, mhz_arr, out_dir)

    print("\n" + "=" * 55)
    print("Done. Plots →", out_dir)
    print("=" * 55)


if __name__ == "__main__":
    main()
