#!/usr/bin/env python3
"""Compare sweep inference vs dataset-ground-truth paths (encode / layout).

Isolates why multifreq sweep compare plots look "all blue" while training
metrics look good: sweep uses marginal z -> generated occ/imp -> layout-z decode;
dataset eval uses true occ/imp (or full encode).

Run:
    .venv/bin/python scratch/diagnose_sweep_vs_dataset.py

Outputs:
    scratch/sweep_vs_dataset_report.json
    scratch/sweep_vs_dataset_summary.txt
    scratch/sweep_vs_dataset_plots/sample_*.png  (optional quick viz)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from repo_paths import REPO_ROOT as PROJECT_ROOT, setup_path

setup_path()

from experiments.exp038_true_multi.codes.dataloader_multifreq import (  # noqa: E402
    create_multifreq_data_loaders,
)
from experiments.exp043.codes.gmax_training_patch import (  # noqa: E402
    apply_gmax_config,
    load_heatmap_stats,
    transform_disk_heatmap,
)
from experiments.exp043.codes.inference_vae import VAEInference, load_experiment_config  # noqa: E402
from src_vae.others.heatmap_gmax_norm import gmax_norm_to_physical, heatmap_fg_threshold  # noqa: E402
from src_vae.others.pi_freq_utils import pi_freq_mhz_to_norm  # noqa: E402

# Defaults aligned with multifreq sweep compare (K=30, 300 MHz, sample_0).
EXPERIMENT_DIR = REPO_ROOT / "experiments/exp043"
CHECKPOINT_PATH = (
    REPO_ROOT / "experiments/exp043/runs/run_20260616T165326Z/checkpoints/last_model.pt"
)
DATA_DIR = REPO_ROOT / "datasets/data_multifreq_gmax"
OUT_JSON = REPO_ROOT / "scratch/sweep_vs_dataset_report.json"
OUT_TXT = REPO_ROOT / "scratch/sweep_vs_dataset_summary.txt"
OUT_PLOT_DIR = REPO_ROOT / "scratch/sweep_vs_dataset_plots"

K_VALUE = 30
TARGET_MHZ = 300.0
PI_REF_MHZ = 200.0
SHARED_TEMP = 1.5
SEED = 42
NUM_DATASET_SAMPLES = 5
MHZ_TOL = 2.0
SAVE_PLOTS = True

SWEEP_NPY = (
    PROJECT_ROOT
    / "experiments/exp043/multifreq_heatmap_sweep_30/freq_300MHz/K30/data_sample_0/heatmap_physical.npy"
)


@dataclass
class PathMetrics:
    path: str
    phys_p50: float
    phys_p95: float
    phys_p99: float
    phys_max: float
    fg_mse_train: float
    pearson_r: float
    mae_phys: float
    occ_match_frac: float | None = None
    imp_mse_log: float | None = None


def _config_ns() -> SimpleNamespace:
    cfg_path = EXPERIMENT_DIR / "config.yaml"
    raw = load_experiment_config(cfg_path) if cfg_path.is_file() else {}
    c = SimpleNamespace(**raw)
    c.data_dir = str(DATA_DIR)
    stats = load_heatmap_stats(DATA_DIR)
    apply_gmax_config(c, stats)
    return c


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _pi_mhz_from_norm(pi_norm: float) -> float:
    """Invert log10 norm → MHz (approx for filtering)."""
    import math

    _LOG10_MIN = math.log10(1e6)
    _LOG10_RANGE = math.log10(600e6) - _LOG10_MIN
    hz = 10 ** (pi_norm * _LOG10_RANGE + _LOG10_MIN)
    return hz / 1e6


def _find_val_samples(
    val_loader,
    *,
    k: int,
    mhz: float,
    max_n: int,
    mhz_tol: float,
) -> list[dict]:
    """Collect up to max_n val batches items matching K and MHz."""
    found: list[dict] = []
    target_norm = pi_freq_mhz_to_norm(mhz)
    for batch in val_loader:
        bsz = batch["K"].shape[0]
        for j in range(bsz):
            kj = int(batch["K"][j].item())
            pi_j = float(batch["PI_freq"][j].item())
            pi_mhz = _pi_mhz_from_norm(pi_j)
            if kj != k:
                continue
            if abs(pi_mhz - mhz) > mhz_tol and abs(pi_j - target_norm) > 0.01:
                continue
            item = {key: (val[j : j + 1] if isinstance(val, torch.Tensor) else val) for key, val in batch.items()}
            found.append(item)
            if len(found) >= max_n:
                return found
    return found


def _fg_mask(hm_train: torch.Tensor, fg_thr: float) -> torch.Tensor:
    return hm_train > fg_thr


def _fg_mse(recon: torch.Tensor, target: torch.Tensor, fg_thr: float) -> float:
    fg = (target > fg_thr).float()
    err = (recon - target).pow(2)
    n = fg.sum().clamp(min=1.0)
    return float((err * fg).sum().item() / n.item())


def _to_phys(hm_train: torch.Tensor, engine: VAEInference) -> np.ndarray:
    phys = engine.denorm_heatmap_physical(hm_train)
    arr = phys.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0, 0]
    elif arr.ndim == 3:
        arr = arr[0]
    return arr.astype(np.float64)


def _gt_phys_from_disk(hm_lin: torch.Tensor, gmax: float) -> np.ndarray:
    lin = hm_lin.detach().cpu().float()
    if lin.dim() == 4:
        lin = lin[0, 0]
    elif lin.dim() == 3:
        lin = lin[0]
    return gmax_norm_to_physical(lin, gmax).cpu().numpy().astype(np.float64)


def _phys_stats(phys: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    fg = phys[mask] if phys.ndim == 2 else phys.reshape(-1)[mask.reshape(-1)]
    if fg.size == 0:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "p50": float(np.percentile(fg, 50)),
        "p95": float(np.percentile(fg, 95)),
        "p99": float(np.percentile(fg, 99)),
        "max": float(fg.max()),
    }


def _pearson_r(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    av = a[mask]
    bv = b[mask]
    if av.size < 2:
        return float("nan")
    if av.std() < 1e-12 or bv.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def _occ_match_frac(gen_occ: torch.Tensor, gt_occ: torch.Tensor, k: int) -> float:
    """Fraction of top-K decap indices that match between gen and GT."""
    g = gen_occ[0].detach().cpu()
    t = gt_occ[0].detach().cpu()
    if k <= 0:
        return 1.0
    gi = set(g.topk(min(k, g.numel())).indices.tolist())
    ti = set(t.topk(min(k, t.numel())).indices.tolist())
    return len(gi & ti) / max(len(gi | ti), 1)


def _imp_mse_log(gen_imp: torch.Tensor, gt_imp: torch.Tensor, log_mean: float, log_std: float) -> float:
    g = gen_imp[0, 0].detach().cpu().numpy()
    t = gt_imp[0, 0].detach().cpu().numpy() if gt_imp.dim() == 3 else gt_imp[0].detach().cpu().numpy()
    return float(np.mean((g - t) ** 2))


@torch.inference_mode()
def _run_path(
    engine: VAEInference,
    model,
    *,
    path: str,
    occ: torch.Tensor,
    imp: torch.Tensor,
    hm_gt_train: torch.Tensor | None,
    k: int,
    mhz: float,
    device: torch.device,
    fg_thr: float,
    mask: np.ndarray,
    gt_phys: np.ndarray,
    gt_occ: torch.Tensor | None = None,
    gt_imp: torch.Tensor | None = None,
) -> PathMetrics:
    if path == "encode_gt":
        if hm_gt_train is None:
            raise ValueError("encode_gt requires hm_gt_train")
        hm_z, occ_out, imp_out = model.inference(
            1,
            device,
            K=k,
            PI_freq=mhz,
            pi_freq_unit="mhz",
            latent_stats=engine.latent_stats,
            per_K_latent_stats=engine.per_K_latent_stats,
            shared_temp=SHARED_TEMP,
            mode="encode",
            heatmap=hm_gt_train,
            occupancy=occ,
            impedance=imp,
        )
    elif path in ("layout_gt", "layout_gt_native_pi", "sweep_anchor_blend"):
        mode = "anchor_blend"
        inf_kw: dict = dict(
            latent_stats=engine.latent_stats,
            per_K_latent_stats=engine.per_K_latent_stats,
            shared_temp=SHARED_TEMP,
            mode=mode,
            pi_ref_mhz=PI_REF_MHZ,
        )
        if path == "layout_gt":
            inf_kw["occupancy"] = occ
            inf_kw["impedance"] = imp
        elif path == "layout_gt_native_pi":
            # Match eval_cross_freq: encode layout at target MHz, not pi_ref.
            z = model.encode_layout_latent(occ, imp, torch.tensor([k], device=device), 
                torch.tensor([pi_freq_mhz_to_norm(mhz)], device=device))
            hm_z = model.decode_heatmap_blended(z, torch.tensor([k], device=device), mhz)
            occ_out, imp_out = occ, imp
            phys = _to_phys(hm_z, engine)
            st = _phys_stats(phys, mask)
            mse = _fg_mse(hm_z, hm_gt_train, fg_thr) if hm_gt_train is not None else float("nan")
            r = _pearson_r(gt_phys, phys, mask)
            mae = float(np.mean(np.abs(gt_phys[mask] - phys[mask])))
            return PathMetrics(
                path=path, phys_p50=st["p50"], phys_p95=st["p95"], phys_p99=st["p99"],
                phys_max=st["max"], fg_mse_train=mse, pearson_r=r, mae_phys=mae,
                occ_match_frac=1.0, imp_mse_log=0.0,
            )
        hm_z, occ_out, imp_out = model.inference(
            1,
            device,
            K=k,
            PI_freq=mhz,
            pi_freq_unit="mhz",
            **inf_kw,
        )
    else:
        raise ValueError(path)

    phys = _to_phys(hm_z, engine)
    st = _phys_stats(phys, mask)
    mse = _fg_mse(hm_z, hm_gt_train, fg_thr) if hm_gt_train is not None else float("nan")
    r = _pearson_r(gt_phys, phys, mask)
    mae = float(np.mean(np.abs(gt_phys[mask] - phys[mask])))

    occ_frac = None
    imp_mse = None
    if gt_occ is not None:
        occ_frac = _occ_match_frac(occ_out, gt_occ, k)
    if gt_imp is not None:
        imp_mse = _imp_mse_log(imp_out, gt_imp, engine.imp_log_mean, engine.imp_log_std)

    return PathMetrics(
        path=path,
        phys_p50=st["p50"],
        phys_p95=st["p95"],
        phys_p99=st["p99"],
        phys_max=st["max"],
        fg_mse_train=mse,
        pearson_r=r,
        mae_phys=mae,
        occ_match_frac=occ_frac,
        imp_mse_log=imp_mse,
    )


def _metrics_from_npy(npy_path: Path, gt_phys: np.ndarray, mask: np.ndarray) -> PathMetrics | None:
    if not npy_path.is_file():
        return None
    phys = np.load(npy_path).astype(np.float64)
    if phys.ndim == 3:
        phys = phys[0]
    st = _phys_stats(phys, mask)
    r = _pearson_r(gt_phys, phys, mask)
    mae = float(np.mean(np.abs(gt_phys[mask] - phys[mask])))
    return PathMetrics(
        path=f"saved_sweep:{npy_path.name}",
        phys_p50=st["p50"],
        phys_p95=st["p95"],
        phys_p99=st["p99"],
        phys_max=st["max"],
        fg_mse_train=float("nan"),
        pearson_r=r,
        mae_phys=mae,
    )


def _save_plot(
    gt_phys: np.ndarray,
    preds: dict[str, np.ndarray],
    mask: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    n = 1 + len(preds)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), layout="constrained")
    if n == 1:
        axes = [axes]
    vmax = float(np.percentile(gt_phys[mask], 99.9)) if mask.any() else float(gt_phys.max())
    vmax = max(vmax, 1e-6)

    def _show(ax, arr, label):
        m = np.ma.masked_where(~mask, arr)
        im = ax.imshow(m, cmap="jet", origin="lower", vmin=0, vmax=vmax)
        ax.set_title(label, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _show(axes[0], gt_phys, "GT (dataset)")
    for ax, (name, arr) in zip(axes[1:], preds.items()):
        _show(ax, arr, name)
    fig.suptitle(title, fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _format_row(m: PathMetrics) -> str:
    occ = f" occ={m.occ_match_frac:.2f}" if m.occ_match_frac is not None else ""
    return (
        f"  {m.path:22s}  p95={m.phys_p95:6.2f}Ω  max={m.phys_max:6.2f}Ω  "
        f"r={m.pearson_r:6.3f}  mae={m.mae_phys:5.3f}  mse_train={m.fg_mse_train:.5f}{occ}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=K_VALUE)
    parser.add_argument("--mhz", type=float, default=TARGET_MHZ)
    parser.add_argument("--num-samples", type=int, default=NUM_DATASET_SAMPLES)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--sweep-npy", type=Path, default=SWEEP_NPY)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c = _config_ns()
    fg_thr = heatmap_fg_threshold(c)
    gmax = float(c.global_max_ohm)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}  K={args.k}  MHz={args.mhz}  pi_ref={PI_REF_MHZ}  seed={SEED}")
    print(f"Train space: {getattr(c, 'heatmap_train_space', 'linear')}  fg_thr={fg_thr:.6f}")

    engine = VAEInference(checkpoint_path=str(args.checkpoint), device=device)
    stats_path = EXPERIMENT_DIR / "metrics/latent_stats.json"
    engine.load_latent_stats(str(stats_path) if stats_path.is_file() else None)
    model = engine.model
    mask = engine.binary_mask.astype(bool)

    _, val_ld = create_multifreq_data_loaders(
        data_dir=str(DATA_DIR),
        batch_size=32,
        num_workers=0,
        train_split=float(getattr(c, "train_split", 0.9)),
        seed=42,
        pin_memory=False,
        split_by_design=getattr(c, "split_by_design", True),
        stratify_by_k=getattr(c, "stratify_by_k", True),
        balance_k=False,
        balance_freq=False,
        cache_in_ram=False,
        val_batch_size=32,
    )

    samples = _find_val_samples(
        val_ld, k=args.k, mhz=args.mhz, max_n=args.num_samples, mhz_tol=MHZ_TOL,
    )
    if not samples:
        raise SystemExit(f"No val samples found for K={args.k} near {args.mhz} MHz")

    print(f"Found {len(samples)} val sample(s) at K={args.k}, ~{args.mhz} MHz\n")

    torch.manual_seed(SEED)
    all_rows: list[dict] = []
    summary_lines: list[str] = []

    for si, batch in enumerate(samples):
        batch = _batch_to_device(batch, device)
        hm_lin = batch["heatmap_norm"]
        if hm_lin.dim() == 3:
            hm_lin = hm_lin.unsqueeze(1)
        occ = batch["occupancy"]
        imp = batch["impedance"]
        if imp.dim() == 2:
            imp = imp.unsqueeze(1)
        elif imp.dim() == 3:
            imp = imp[:, :1]
        hm_gt_train = transform_disk_heatmap(hm_lin, c)
        gt_phys = _gt_phys_from_disk(hm_lin, gmax)
        k = int(batch["K"][0].item())

        paths = ("sweep_anchor_blend", "layout_gt", "layout_gt_native_pi", "encode_gt")
        metrics: list[PathMetrics] = []
        pred_phys: dict[str, np.ndarray] = {}

        for path in paths:
            m = _run_path(
                engine,
                model,
                path=path,
                occ=occ,
                imp=imp,
                hm_gt_train=hm_gt_train,
                k=k,
                mhz=args.mhz,
                device=device,
                fg_thr=fg_thr,
                mask=mask,
                gt_phys=gt_phys,
                gt_occ=occ,
                gt_imp=imp,
            )
            metrics.append(m)

        if si == 0 and not args.no_plots:
            torch.manual_seed(SEED)
            for path in paths:
                if path == "encode_gt":
                    hm_z, _, _ = model.inference(
                        1, device, K=k, PI_freq=args.mhz, pi_freq_unit="mhz",
                        latent_stats=engine.latent_stats,
                        per_K_latent_stats=engine.per_K_latent_stats,
                        shared_temp=SHARED_TEMP, mode="encode",
                        heatmap=hm_gt_train, occupancy=occ, impedance=imp,
                    )
                else:
                    inf_kw = dict(
                        latent_stats=engine.latent_stats,
                        per_K_latent_stats=engine.per_K_latent_stats,
                        shared_temp=SHARED_TEMP,
                        mode="anchor_blend",
                        pi_ref_mhz=PI_REF_MHZ,
                    )
                    if path == "layout_gt":
                        inf_kw["occupancy"] = occ
                        inf_kw["impedance"] = imp
                    hm_z, _, _ = model.inference(
                        1, device, K=k, PI_freq=args.mhz, pi_freq_unit="mhz", **inf_kw,
                    )
                pred_phys[path] = _to_phys(hm_z, engine)
            if args.sweep_npy.is_file():
                saved_arr = np.load(args.sweep_npy).astype(np.float64)
                if saved_arr.ndim == 3:
                    saved_arr = saved_arr[0]
                pred_phys["saved_sweep_npy"] = saved_arr
            _save_plot(
                gt_phys,
                pred_phys,
                mask,
                OUT_PLOT_DIR / f"sample_{si}_K{k}_{int(args.mhz)}MHz.png",
                f"K={k} {args.mhz} MHz — sweep vs dataset paths",
            )

        block = f"--- val sample {si} (K={k}) ---"
        print(block)
        summary_lines.append(block)
        for m in metrics:
            line = _format_row(m)
            print(line)
            summary_lines.append(line)
            all_rows.append({"sample_idx": si, **asdict(m)})

        if si == 0:
            saved = _metrics_from_npy(args.sweep_npy, gt_phys, mask)
            if saved:
                note = (
                    "  (saved sweep npy is vs this dataset GT — compare uses ECADStar sim, "
                    "not this GT heatmap)"
                )
                print(_format_row(saved))
                print(note)
                summary_lines.append(_format_row(saved))
                summary_lines.append(note)
                all_rows.append({"sample_idx": si, **asdict(saved), "note": "vs dataset GT only"})

    # Aggregate means per path
    print("\n=== MEAN over dataset samples ===")
    summary_lines.append("\n=== MEAN over dataset samples ===")
    for path in ("sweep_anchor_blend", "layout_gt", "layout_gt_native_pi", "encode_gt"):
        subset = [r for r in all_rows if r.get("path") == path]
        if not subset:
            continue
        mean_r = float(np.nanmean([r["pearson_r"] for r in subset]))
        mean_p95 = float(np.mean([r["phys_p95"] for r in subset]))
        mean_mse = float(np.nanmean([r["fg_mse_train"] for r in subset]))
        line = f"  {path:22s}  mean_r={mean_r:.3f}  mean_p95={mean_p95:.2f}Ω  mean_mse_train={mean_mse:.5f}"
        print(line)
        summary_lines.append(line)

    verdict = (
        "\nInterpretation:\n"
        "  • encode_gt  ≈ upper bound (model sees GT heatmap).\n"
        "  • layout_gt_native_pi ≈ eval_cross_freq (encode layout at target MHz).\n"
        "  • layout_gt  ≈ sweep layout path (encode at pi_ref=200, decode at target).\n"
        "  • sweep_anchor_blend ≈ compare pipeline (generated occ/imp from marginal z).\n"
        "  If layout_gt ≫ sweep but encode ≈ layout_gt, bottleneck is generated layout not heatmap decoder.\n"
    )
    print(verdict)
    summary_lines.append(verdict)

    payload = {
        "checkpoint": str(args.checkpoint),
        "k": args.k,
        "mhz": args.mhz,
        "pi_ref_mhz": PI_REF_MHZ,
        "seed": SEED,
        "shared_temp": SHARED_TEMP,
        "rows": all_rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    OUT_TXT.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {OUT_TXT}")
    if not args.no_plots:
        print(f"Plots: {OUT_PLOT_DIR}/")


if __name__ == "__main__":
    main()
