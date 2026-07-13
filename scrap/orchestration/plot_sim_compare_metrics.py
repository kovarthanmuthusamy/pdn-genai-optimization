"""Plot sim_compare_metrics.json from a multifreq sweep run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_METRICS = (
    _REPO
    / "experiments/exp050/multifreq_heatmap_sweep_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20"
    / "sim_compare_metrics.json"
)


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text())


def _pivot(rows: list[dict], field: str) -> tuple[list[int], list[float], np.ndarray]:
    mhz_vals = sorted({int(r["mhz"]) for r in rows})
    k_vals = sorted({int(r["k"]) for r in rows})
    grid = np.full((len(mhz_vals), len(k_vals)), np.nan)
    lookup = {(int(r["mhz"]), int(r["k"])): float(r[field]) for r in rows}
    for i, m in enumerate(mhz_vals):
        for j, k in enumerate(k_vals):
            grid[i, j] = lookup.get((m, k), np.nan)
    return k_vals, mhz_vals, grid


def plot_pearson_vs_k(rows: list[dict], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    mhz_vals = sorted({int(r["mhz"]) for r in rows})
    cmap = plt.cm.viridis(np.linspace(0.1, 0.95, len(mhz_vals)))

    for color, m in zip(cmap, mhz_vals):
        sub = sorted([r for r in rows if int(r["mhz"]) == m], key=lambda r: int(r["k"]))
        ks = [int(r["k"]) for r in sub]
        rs = [float(r["pearson_r"]) for r in sub]
        ax.plot(ks, rs, "o-", color=color, label=f"{m} MHz", linewidth=1.8, markersize=5)

    ax.axhline(0.85, color="0.55", linestyle="--", linewidth=1, label="r = 0.85")
    ax.axhline(0.5, color="0.75", linestyle=":", linewidth=1, label="r = 0.50")
    ax.set_xlabel("K (val layout id)")
    ax.set_ylabel("Pearson r (sim real vs generated)")
    ax.set_title("exp050 sim-real QC — spatial correlation by layout K")
    ax.set_xticks(sorted({int(r["k"]) for r in rows}))
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", ncol=2, fontsize=8)
    fig.tight_layout()
    path = out_dir / "sim_metrics_pearson_vs_k.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_max_ratio_vs_k(rows: list[dict], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    mhz_vals = sorted({int(r["mhz"]) for r in rows})
    cmap = plt.cm.plasma(np.linspace(0.15, 0.9, len(mhz_vals)))

    for color, m in zip(cmap, mhz_vals):
        sub = sorted([r for r in rows if int(r["mhz"]) == m], key=lambda r: int(r["k"]))
        ks = [int(r["k"]) for r in sub]
        ratios = [float(r["max_ratio"]) for r in sub]
        ax.plot(ks, ratios, "o-", color=color, label=f"{m} MHz", linewidth=1.8, markersize=5)

    ax.axhline(1.0, color="0.35", linestyle="-", linewidth=1.2, label="ratio = 1.0")
    ax.axhspan(0.5, 2.0, color="0.9", alpha=0.35, label="acceptable band")
    ax.set_xlabel("K (val layout id)")
    ax.set_ylabel("max_ratio (gen_max / real_max)")
    ax.set_title("exp050 sim-real QC — peak magnitude ratio by layout K")
    ax.set_xticks(sorted({int(r["k"]) for r in rows}))
    ax.set_ylim(0, max(7.5, max(float(r["max_ratio"]) for r in rows) * 1.05))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    path = out_dir / "sim_metrics_max_ratio_vs_k.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_heatmap(rows: list[dict], field: str, title: str, out_name: str, out_dir: Path, vmin=None, vmax=None) -> Path:
    k_vals, mhz_vals, grid = _pivot(rows, field)
    fig, ax = plt.subplots(figsize=(12, 4.2))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(k_vals)))
    ax.set_xticklabels(k_vals)
    ax.set_yticks(range(len(mhz_vals)))
    ax.set_yticklabels([f"{m} MHz" for m in mhz_vals])
    ax.set_xlabel("K")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(field)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            val = grid[i, j]
            if np.isnan(val):
                continue
            text_color = "white" if val < (vmin or 0) + 0.35 * ((vmax or 1) - (vmin or 0)) else "black"
            if field == "pearson_r" and val < 0.5:
                text_color = "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=text_color)
    fig.tight_layout()
    path = out_dir / out_name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_mhz_summary(by_mhz: list[dict], out_dir: Path) -> Path:
    mhz_vals = [int(r["mhz"]) for r in by_mhz]
    pearson = [float(r["pearson_r_mean"]) for r in by_mhz]
    mae = [float(r["mae_ohm_mean"]) for r in by_mhz]
    ratio = [float(r["max_ratio_mean"]) for r in by_mhz]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

    axes[0].bar([str(m) for m in mhz_vals], pearson, color=colors)
    axes[0].axhline(0.85, color="0.5", linestyle="--", linewidth=1)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Mean Pearson r")
    axes[0].set_xlabel("MHz")

    axes[1].bar([str(m) for m in mhz_vals], mae, color=colors)
    axes[1].set_title("Mean MAE (Ω)")
    axes[1].set_xlabel("MHz")

    axes[2].bar([str(m) for m in mhz_vals], ratio, color=colors)
    axes[2].axhline(1.0, color="0.35", linestyle="-", linewidth=1)
    axes[2].set_title("Mean max_ratio")
    axes[2].set_xlabel("MHz")

    fig.suptitle("exp050 sim-real QC — mean metrics across K layouts", y=1.02)
    fig.tight_layout()
    path = out_dir / "sim_metrics_mhz_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_by_k_bars(by_k: list[dict], out_dir: Path) -> Path:
    by_k = sorted(by_k, key=lambda r: int(r["k"]))
    ks = [int(r["k"]) for r in by_k]
    pearson = [float(r["pearson_r_mean"]) for r in by_k]
    colors = ["#C44E52" if p < 0.65 else "#DD8452" if p < 0.8 else "#55A868" for p in pearson]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar([str(k) for k in ks], pearson, color=colors)
    ax.axhline(0.85, color="0.45", linestyle="--", linewidth=1, label="r = 0.85")
    ax.set_xlabel("K")
    ax.set_ylabel("Mean Pearson r (all MHz)")
    ax.set_title("exp050 sim-real QC — per-layout K quality (mean over 5 frequencies)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "sim_metrics_by_k_mean_pearson.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sim_compare_metrics.json")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    metrics_path = args.metrics.resolve()
    out_dir = (args.out_dir or metrics_path.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_metrics(metrics_path)
    rows = data["per_sample"]

    paths = [
        plot_pearson_vs_k(rows, out_dir),
        plot_max_ratio_vs_k(rows, out_dir),
        plot_heatmap(
            rows,
            "pearson_r",
            "Pearson r heatmap (sim real vs generated)",
            "sim_metrics_pearson_heatmap.png",
            out_dir,
            vmin=0,
            vmax=1,
        ),
        plot_heatmap(
            rows,
            "max_ratio",
            "max_ratio heatmap (gen_max / real_max)",
            "sim_metrics_max_ratio_heatmap.png",
            out_dir,
            vmin=0,
            vmax=3,
        ),
        plot_mhz_summary(data["by_mhz"], out_dir),
        plot_by_k_bars(data["by_k"], out_dir),
    ]

    print(f"Wrote {len(paths)} plots to {out_dir}")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
