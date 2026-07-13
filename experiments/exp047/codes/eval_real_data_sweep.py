"""Evaluate exp046 heatmap quality using REAL dataset layouts.

Loads real (occ, imp, heatmap) from val split and evaluates:
  1. encode  : encode(hm, occ, imp, K, pi) → decode → compare with real hm
  2. layout  : encode_layout_latent(occ, imp, K, pi) → decode → compare with real hm

Reports per-frequency: MAE, FG-MSE, Pearson r, peak-loc error, p95, p99.9, max.
Saves side-by-side comparison images (real vs generated).

Run:
  cd /home/ubuntu/genai_pdn
  python -m experiments.exp047.codes.eval_real_data_sweep
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
from pathlib import Path

_REPO_BOOT = Path(__file__).resolve().parents[3]
if str(_REPO_BOOT) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOT))

from repo_paths import REPO_ROOT as PROJECT_ROOT, setup_path
setup_path()

from experiments.exp047.codes.exp047_eval_common import (
    load_exp_config,
    load_model,
    pi_norm_to_mhz,
    resolve_paths,
)
from experiments.exp047.codes.spatial_metrics import peak_loc_err, pearson_fg
from experiments.exp038_true_multi.codes.dataloader_multifreq import (
    create_multifreq_data_loaders,
)
from src_vae.others.heatmap_z_clip import heatmap_z_to_physical

# ── Config ────────────────────────────────────────────────────────────────────
EVAL_MHZ: list[float] = [10.0, 63.0, 300.0, 330.0, 400.0]
MAX_SAMPLES_PER_MHZ = 8
MAX_BATCHES = 0           # 0 = full val loader
NUM_PLOT_SAMPLES = 2      # side-by-side images per frequency
CHECKPOINT: str | None = None  # None = last_model.pt

OUTPUT_DIR_NAME = "eval_real_data_sweep"
# ──────────────────────────────────────────────────────────────────────────────


def _fg_mse(recon: torch.Tensor, target: torch.Tensor, bg: float, margin: float = 0.5) -> torch.Tensor:
    fg = (target > bg + margin).float()
    err = (recon - target).pow(2)
    n = fg.sum(dim=(1, 2, 3)).clamp(min=1.0)
    return (err * fg).sum(dim=(1, 2, 3)) / n


@torch.inference_mode()
def evaluate(
    model,
    val_loader: DataLoader,
    *,
    bg: float,
    eval_mhz: list[float],
    log_mean: float,
    log_std: float,
    clip_lo: float | None,
    clip_hi: float | None,
    mask: np.ndarray,
    max_samples: int,
    max_batches: int,
    device: torch.device,
) -> dict:
    """Run eval on val set, return per-(mhz, mode) stats + sample data for plots."""
    model.eval()
    base = getattr(model, "_orig_mod", model)
    mhz_tol = 2.0

    buckets: dict[tuple[str, float], dict[str, list]] = {}
    plot_samples: dict[tuple[str, float], list[dict]] = {}

    def _init_bucket(key):
        if key not in buckets:
            buckets[key] = {
                "mse": [], "mae": [], "pearson": [], "peak_loc": [],
                "real_max": [], "gen_max": [],
                "real_p95": [], "gen_p95": [],
                "real_p999": [], "gen_p999": [],
            }
            plot_samples[key] = []

    for bi, batch in enumerate(val_loader):
        if 0 < max_batches <= bi:
            break

        hm = batch["heatmap_norm"].to(device)
        occ = batch["occupancy"].to(device)
        imp = batch["impedance"].to(device)
        K = batch["K"].to(device)
        pi_native = batch["PI_freq"].to(device)
        if hm.dim() == 3:
            hm = hm.unsqueeze(1)
        if imp.dim() == 2:
            imp = imp.unsqueeze(1)
        elif imp.dim() == 3:
            imp = imp[:, :1]

        pi_mhz = pi_norm_to_mhz(pi_native.cpu().numpy())
        hm_enc = hm.masked_fill(hm < bg, 0.0)

        for target_mhz in eval_mhz:
            idx_mask = np.abs(pi_mhz - target_mhz) < mhz_tol
            if not idx_mask.any():
                continue

            sel = torch.where(torch.tensor(idx_mask, device=device))[0]
            hm_sel = hm[sel]
            hm_enc_sel = hm_enc[sel]
            occ_sel = occ[sel]
            imp_sel = imp[sel]
            K_sel = K[sel]
            pi_sel = pi_native[sel]

            z_enc, _, _, _ = base.encode(hm_enc_sel, occ_sel, imp_sel, K_sel, pi_sel)
            skips_enc = base._last_heatmap_skips
            z_lay = base.encode_layout_latent(occ_sel, imp_sel, K_sel, pi_sel)

            rh_enc, _, _ = base.decode(z_enc, K_sel, pi_sel, occupancy=occ_sel, heatmap_skips=skips_enc)
            rh_lay, _, _ = base.decode(z_lay, K_sel, pi_sel, occupancy=occ_sel)

            hm_real_phys = heatmap_z_to_physical(hm_sel, log_mean, log_std, clip_lo=clip_lo, clip_hi=clip_hi)

            for kind, recon in (("encode", rh_enc), ("layout", rh_lay)):
                key = (kind, target_mhz)
                _init_bucket(key)
                b = buckets[key]
                if len(b["mse"]) >= max_samples:
                    continue

                mse_vals = _fg_mse(recon, hm_sel, bg)
                pr_vals = pearson_fg(recon, hm_sel, bg)
                pl_vals = peak_loc_err(recon, hm_sel, bg)

                recon_phys = heatmap_z_to_physical(recon, log_mean, log_std, clip_lo=clip_lo, clip_hi=clip_hi)

                for j in range(min(sel.shape[0], max_samples - len(b["mse"]))):
                    b["mse"].append(float(mse_vals[j]))
                    b["pearson"].append(float(pr_vals[j]))
                    b["peak_loc"].append(float(pl_vals[j]))

                    r_np = hm_real_phys[j, 0].cpu().numpy()
                    g_np = recon_phys[j, 0].cpu().numpy()
                    r_fg = r_np[mask]
                    g_fg = g_np[mask]

                    mae = float(np.mean(np.abs(r_fg - g_fg))) if r_fg.size else 0.0
                    b["mae"].append(mae)
                    b["real_max"].append(float(r_fg.max()) if r_fg.size else 0.0)
                    b["gen_max"].append(float(g_fg.max()) if g_fg.size else 0.0)
                    b["real_p95"].append(float(np.percentile(r_fg, 95)) if r_fg.size else 0.0)
                    b["gen_p95"].append(float(np.percentile(g_fg, 95)) if g_fg.size else 0.0)
                    b["real_p999"].append(float(np.percentile(r_fg, 99.9)) if r_fg.size else 0.0)
                    b["gen_p999"].append(float(np.percentile(g_fg, 99.9)) if g_fg.size else 0.0)

                    if len(plot_samples[key]) < NUM_PLOT_SAMPLES:
                        plot_samples[key].append({
                            "real": r_np.copy(),
                            "gen": g_np.copy(),
                            "K": int(K_sel[j].item()),
                        })

    return {"buckets": buckets, "plot_samples": plot_samples}


def _summarise(buckets: dict) -> list[dict]:
    rows = []
    for (kind, mhz), vals in sorted(buckets.items()):
        n = len(vals["mse"])
        if n == 0:
            continue
        rows.append({
            "mode": kind,
            "mhz": mhz,
            "n": n,
            "fg_mse": np.mean(vals["mse"]),
            "mae_phys": np.mean(vals["mae"]),
            "pearson_r": np.mean(vals["pearson"]),
            "peak_loc_err": np.mean(vals["peak_loc"]),
            "real_max": np.mean(vals["real_max"]),
            "gen_max": np.mean(vals["gen_max"]),
            "real_p95": np.mean(vals["real_p95"]),
            "gen_p95": np.mean(vals["gen_p95"]),
            "real_p999": np.mean(vals["real_p999"]),
            "gen_p999": np.mean(vals["gen_p999"]),
        })
    return rows


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV → {path}")


def _save_plots(
    plot_samples: dict,
    mask: np.ndarray,
    out_dir: Path,
) -> None:
    cmap = mpl.colormaps["jet"].resampled(22).copy()
    cmap.set_bad("white")

    for (kind, mhz), samples in sorted(plot_samples.items()):
        if not samples:
            continue
        n = len(samples)
        fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n), squeeze=False)

        for i, s in enumerate(samples):
            real = s["real"]
            gen = s["gen"]
            diff = np.abs(real - gen)

            real_m = np.ma.masked_where(~mask, real)
            gen_m = np.ma.masked_where(~mask, gen)
            diff_m = np.ma.masked_where(~mask, diff)

            r_fg = real_m.compressed()
            g_fg = gen_m.compressed()
            vmax_shared = max(float(r_fg.max()), float(g_fg.max()), 1e-6)
            pearson = float(np.corrcoef(r_fg, g_fg)[0, 1]) if len(r_fg) > 1 else float("nan")

            ax0 = axes[i, 0]
            im0 = ax0.imshow(real_m, cmap=cmap, interpolation="bicubic",
                             aspect="auto", origin="lower", vmin=0, vmax=vmax_shared)
            ax0.set_title(f"Real  max={float(r_fg.max()):.2f}  p95={float(np.percentile(r_fg, 95)):.2f}", fontsize=10)
            fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

            ax1 = axes[i, 1]
            im1 = ax1.imshow(gen_m, cmap=cmap, interpolation="bicubic",
                             aspect="auto", origin="lower", vmin=0, vmax=vmax_shared)
            ax1.set_title(f"Gen   max={float(g_fg.max()):.2f}  p95={float(np.percentile(g_fg, 95)):.2f}", fontsize=10)
            fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

            ax2 = axes[i, 2]
            im2 = ax2.imshow(diff_m, cmap="hot", interpolation="bicubic",
                             aspect="auto", origin="lower", vmin=0, vmax=max(float(diff_m.max()), 0.1))
            ax2.set_title(f"|Diff| r={pearson:.3f}  K={s['K']}", fontsize=10)
            fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        fig.suptitle(f"{kind.upper()} mode @ {mhz:.0f} MHz  (real dataset val samples)", fontsize=14, y=1.01)
        fig.tight_layout()
        fname = out_dir / f"{kind}_{int(mhz)}MHz.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot → {fname}")


def main() -> None:
    cfg = load_exp_config()
    paths = resolve_paths(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(CHECKPOINT) if CHECKPOINT else paths["checkpoint"]
    model, model_cfg, _, _ = load_model(ckpt_path, device)

    data_dir = str(paths["data_dir"])
    mask = np.load(paths["binary_mask"]).astype(bool)
    ns = json.loads(Path(paths["norm_stats"]).read_text(encoding="utf-8"))
    hm = ns["Heatmap"]
    log_mean, log_std = hm["log_mean"], hm["log_std"]
    clip_lo = float(hm["clip_min"]) if hm.get("clip_min") is not None else None
    clip_hi = float(hm["clip_max"]) if hm.get("clip_max") is not None else None
    bg = float(paths["background_value"])

    _, val_loader = create_multifreq_data_loaders(
        data_dir=data_dir,
        batch_size=64,
        num_workers=4,
        train_split=float(cfg.get("train_split", 0.9)),
        seed=42,
        pin_memory=device.type == "cuda",
        split_by_design=cfg.get("split_by_design", True),
        stratify_by_k=cfg.get("stratify_by_k", True),
        balance_k=False,
        balance_freq=False,
        cache_in_ram=cfg.get("cache_in_ram", True),
    )

    print(f"\nEvaluating on val split at {EVAL_MHZ} MHz")
    print(f"  data_dir   = {data_dir}")
    print(f"  checkpoint = {ckpt_path}")
    print(f"  device     = {device}")
    print(f"  max_samples/freq = {MAX_SAMPLES_PER_MHZ}")
    print(f"  denorm: log_mean={log_mean:.4f} log_std={log_std:.4f} clip=[{clip_lo}, {clip_hi}]")
    print()

    result = evaluate(
        model, val_loader,
        bg=bg,
        eval_mhz=EVAL_MHZ,
        log_mean=log_mean,
        log_std=log_std,
        clip_lo=clip_lo,
        clip_hi=clip_hi,
        mask=mask,
        max_samples=MAX_SAMPLES_PER_MHZ,
        max_batches=MAX_BATCHES,
        device=device,
    )

    rows = _summarise(result["buckets"])

    exp_dir = Path(__file__).resolve().parents[1]
    out_dir = exp_dir / OUTPUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(rows, out_dir / "eval_summary.csv")
    _save_plots(result["plot_samples"], mask, out_dir)

    print("\n" + "=" * 90)
    print(f"{'Mode':<10} {'MHz':>6} {'n':>4}  {'FG-MSE':>8} {'MAE(Ω)':>8} "
          f"{'r':>6} {'peakΔ':>6}  {'R_max':>7} {'G_max':>7} "
          f"{'R_p95':>7} {'G_p95':>7} {'R_p999':>7} {'G_p999':>7}")
    print("-" * 90)
    for r in rows:
        print(f"{r['mode']:<10} {r['mhz']:>6.0f} {r['n']:>4}  "
              f"{r['fg_mse']:>8.4f} {r['mae_phys']:>8.4f} "
              f"{r['pearson_r']:>6.3f} {r['peak_loc_err']:>6.3f}  "
              f"{r['real_max']:>7.2f} {r['gen_max']:>7.2f} "
              f"{r['real_p95']:>7.2f} {r['gen_p95']:>7.2f} "
              f"{r['real_p999']:>7.2f} {r['gen_p999']:>7.2f}")
    print("=" * 90)
    print(f"\nResults → {out_dir}")


if __name__ == "__main__":
    main()
