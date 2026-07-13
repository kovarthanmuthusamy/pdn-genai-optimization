"""
evaluate_vae.py — Post-training evaluation for exp045 (PI_freq-conditioned VAE).

Diagnostics:
  1. Variation within K at several PI_freq (MHz)
  2. Nearest-neighbour distance (generated vs training)
  3. Latent interpolation (fixed K + PI_freq)
  4. Prior / layout sampling sanity
  5. Cross-frequency decode: same layout z, native vs off-anchor MHz
  6. Off-anchor val metrics (encode_cross vs layout_cross) via eval_cross_freq

Usage:
    python experiments/exp045/codes/evaluate_vae.py
    python experiments/exp045/codes/evaluate_vae.py --ckpt checkpoints/last_model.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from torch.utils.data import DataLoader

_CODES = Path(__file__).resolve().parent
_ROOT = _CODES.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_CODES) not in sys.path:
    sys.path.insert(0, str(_CODES))

from exp045_eval_common import (  # noqa: E402
    load_exp_config,
    load_model,
    pi_tensor_mhz,
    resolve_paths,
)
from experiments.exp038_true_multi.codes.dataloader_multifreq import create_multifreq_data_loaders  # noqa: E402
from experiments.exp038_true_multi.codes.eval_cross_freq import run_off_anchor_eval  # noqa: E402
from src_vae.others.dataloader import VAEDataset  # noqa: E402
from src_vae.others.pi_freq_utils import pi_freq_mhz_to_norm  # noqa: E402

K_TEST_PRESENT = 26
K_TEST_RARE = 2
TEST_MHZ = (63.0, 200.0, 400.0)
OFF_ANCHOR_MHZ = (100.0, 175.0, 350.0)
N_VARIATION = 50
N_NN_TRAIN = 500
N_PRIOR = 200
INTERP_STEPS = 8
TEMPS = [0.6, 0.8, 1.0, 1.2, 1.5]


def test_variation(model, latent_stats, device, out_dir: Path, mhz_list: tuple[float, ...]):
    print("\n" + "=" * 60)
    print("TEST 1 — Variation within K at multiple PI_freq (MHz)")
    print("=" * 60)
    results = {}
    for K_val in [K_TEST_PRESENT, K_TEST_RARE]:
        results[K_val] = {}
        for mhz in mhz_list:
            row = []
            for temp in TEMPS:
                hm_t, _, imp_t = model.inference(
                    N_VARIATION, device, K=K_val, PI_freq=mhz, pi_freq_unit="mhz",
                    latent_stats=latent_stats, shared_temp=temp, mode="layout",
                )
                hm = hm_t.cpu().numpy()
                imp = imp_t[:, 0].cpu().numpy()
                hm_std = hm.std(axis=0).mean()
                imp_std = imp.std(axis=0).mean()
                row.append((temp, hm_std, imp_std))
            results[K_val][mhz] = row
            t1 = next(r for r in row if r[0] == 1.0)
            print(f"  K={K_val}  {mhz:.0f} MHz  temp=1.0  hm_std={t1[1]:.5f}  imp_std={t1[2]:.5f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, idx, ylab in zip(axes, [1, 2], ["Heatmap pixel std", "Impedance std"]):
        for K_val in [K_TEST_PRESENT, K_TEST_RARE]:
            for mhz in mhz_list:
                row = results[K_val][mhz]
                ax.plot([r[0] for r in row], [r[idx] for r in row], marker="o",
                        label=f"K={K_val} {mhz:.0f}MHz")
        ax.set_xlabel("Temperature")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "test1_variation_vs_temperature.png", dpi=150)
    plt.close(fig)
    print("  → test1_variation_vs_temperature.png")
    return results


def test_nn_distance(model, latent_stats, device, data_dir, out_dir: Path, mhz: float = 200.0):
    print("\n" + "=" * 60)
    print("TEST 2 — Nearest-neighbour distance")
    print("=" * 60)
    ds = VAEDataset(data_dir=str(data_dir))
    indices = np.random.choice(len(ds), min(N_NN_TRAIN, len(ds)), replace=False)
    train_hm, train_imp = [], []
    for idx in indices:
        s = ds[idx]
        train_hm.append(s["heatmap_norm"].float().numpy().flatten())
        train_imp.append(s["impedance"].float().numpy()[0])
    train_hm = np.stack(train_hm)
    train_imp = np.stack(train_imp)

    results = {}
    for K_val in [K_TEST_PRESENT, K_TEST_RARE]:
        hm_gen, _, imp_gen = model.inference(
            50, device, K=K_val, PI_freq=mhz, pi_freq_unit="mhz",
            latent_stats=latent_stats, shared_temp=1.0, mode="layout",
        )
        hm_g = hm_gen.cpu().numpy().reshape(50, -1)
        imp_g = imp_gen[:, 0].cpu().numpy()
        hm_d = [np.linalg.norm(train_hm - hm_g[i], axis=1).min() for i in range(50)]
        imp_d = [np.linalg.norm(train_imp - imp_g[i], axis=1).min() for i in range(50)]
        results[K_val] = {"hm": np.array(hm_d), "imp": np.array(imp_d)}
        print(f"  K={K_val} @ {mhz:.0f} MHz  hm_NN mean={np.mean(hm_d):.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, K_val in enumerate([K_TEST_PRESENT, K_TEST_RARE]):
        for row, key in enumerate(["hm", "imp"]):
            ax = axes[row, col]
            ax.hist(results[K_val][key], bins=20, color="steelblue", alpha=0.85)
            ax.axvline(results[K_val][key].mean(), color="red", ls="--")
            ax.set_title(f"{key} NN  K={K_val}  @ {mhz:.0f} MHz")
    fig.tight_layout()
    fig.savefig(out_dir / "test2_nn_distance.png", dpi=150)
    plt.close(fig)
    print("  → test2_nn_distance.png")
    return results


def test_interpolation(model, device, data_dir, paths, mhz: float, out_dir: Path):
    print("\n" + "=" * 60)
    print(f"TEST 3 — Latent interpolation  (K={K_TEST_PRESENT}, PI={mhz:.0f} MHz)")
    print("=" * 60)
    ds = VAEDataset(data_dir=str(data_dir))
    idxA = idxB = None
    for i in range(len(ds)):
        if int(ds[i]["K"]) == K_TEST_PRESENT:
            if idxA is None:
                idxA = i
            elif idxB is None:
                idxB = i
                break
    if idxA is None or idxB is None:
        print("  Could not find pair with target K; skipping.")
        return

    def _batch(s):
        return {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v for k, v in s.items()}

    sA, sB = ds[idxA], ds[idxB]
    pi = pi_tensor_mhz(mhz, 1, device)
    with torch.no_grad():
        _, muA, _, _ = model.encode(
            _batch(sA)["heatmap_norm"], _batch(sA)["occupancy"],
            _batch(sA)["impedance"], _batch(sA)["K"], pi,
        )
        _, muB, _, _ = model.encode(
            _batch(sB)["heatmap_norm"], _batch(sB)["occupancy"],
            _batch(sB)["impedance"], _batch(sB)["K"], pi,
        )

    norm_stats = json.load(open(paths["norm_stats"], encoding="utf-8"))
    hm_log_mean, hm_log_std = norm_stats["Heatmap"]["log_mean"], norm_stats["Heatmap"]["log_std"]
    imp_log_mean, imp_log_std = norm_stats["Impedance"]["log_mean"], norm_stats["Impedance"]["log_std"]
    freq = np.load(paths["freq_hz"]).squeeze()
    binary_mask = np.load(paths["binary_mask"])
    target_imp = np.load(paths["target_imp"]).squeeze()

    K_tensor = torch.full((1,), K_TEST_PRESENT, dtype=torch.long, device=device)
    alphas = np.linspace(0, 1, INTERP_STEPS)
    hm_frames, imp_frames = [], []
    with torch.no_grad():
        for alpha in alphas:
            z = (1.0 - alpha) * muA + alpha * muB
            hm, _, imp = model.decode(z, K_tensor, pi)
            hm_phys = (torch.exp(hm * hm_log_std + hm_log_mean) - 1.0).clamp(min=0.0)
            hm_frames.append(hm_phys[0, 0].cpu().numpy())
            imp_frames.append(imp[0, 0].cpu().numpy())

    fig = plt.figure(figsize=(INTERP_STEPS * 3, 7))
    gs = gridspec.GridSpec(2, INTERP_STEPS, hspace=0.35, wspace=0.1)
    vmin, vmax = min(f.min() for f in hm_frames), max(f.max() for f in hm_frames)
    for j, alpha in enumerate(alphas):
        ax_hm = fig.add_subplot(gs[0, j])
        ax_hm.imshow(np.ma.masked_where(~binary_mask, hm_frames[j]), cmap="jet",
                     vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
        ax_hm.set_title(f"α={alpha:.2f}", fontsize=8)
        ax_hm.axis("off")
        ax_imp = fig.add_subplot(gs[1, j])
        ax_imp.loglog(freq, target_imp, "--", color="red", alpha=0.5)
        ax_imp.loglog(freq, np.exp(imp_frames[j] * imp_log_std + imp_log_mean), "-", color="royalblue")
        ax_imp.set_xticks([])
        ax_imp.set_yticks([])
    fig.suptitle(f"Interpolation K={K_TEST_PRESENT}  PI={mhz:.0f} MHz", fontweight="bold")
    fig.savefig(out_dir / f"test3_interpolation_{int(mhz)}MHz.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → test3_interpolation_{int(mhz)}MHz.png")


def test_prior_sampling(model, latent_stats, device, paths, mhz: float, out_dir: Path):
    print("\n" + "=" * 60)
    print(f"TEST 4 — Prior / layout sampling  (@ {mhz:.0f} MHz)")
    print("=" * 60)
    norm_stats = json.load(open(paths["norm_stats"], encoding="utf-8"))
    hm_log_mean, hm_log_std = norm_stats["Heatmap"]["log_mean"], norm_stats["Heatmap"]["log_std"]
    pi = pi_tensor_mhz(mhz, N_PRIOR, device)
    results = {}
    for K_val in [K_TEST_PRESENT, K_TEST_RARE]:
        z_prior = torch.randn(N_PRIOR, model.latent_dim, device=device)
        K_t = torch.full((N_PRIOR,), K_val, dtype=torch.long, device=device)
        with torch.no_grad():
            hm, _, _ = model.decode(z_prior, K_t, pi)
        hm_phys = (torch.exp(hm * hm_log_std + hm_log_mean) - 1.0).clamp(min=0.0)
        hm_std = hm_phys[:, 0].cpu().numpy().std()
        hm_ls, _, _ = model.inference(
            N_PRIOR, device, K=K_val, PI_freq=mhz, pi_freq_unit="mhz",
            latent_stats=latent_stats, mode="layout",
        )
        hm_ls_std = (torch.exp(hm_ls * hm_log_std + hm_log_mean) - 1.0).clamp(min=0.0)[:, 0].cpu().numpy().std()
        degenerate = hm_std < 0.001
        results[K_val] = {"prior_hm_std": hm_std, "layout_hm_std": hm_ls_std, "degenerate": degenerate}
        print(f"  K={K_val}  prior_std={hm_std:.5f}  layout_std={hm_ls_std:.5f}  "
              f"{'DEGENERATE' if degenerate else 'OK'}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, K_val in zip(axes, [K_TEST_PRESENT, K_TEST_RARE]):
        z_p = torch.randn(N_PRIOR, model.latent_dim, device=device)
        K_t = torch.full((N_PRIOR,), K_val, dtype=torch.long, device=device)
        with torch.no_grad():
            hm_p, _, _ = model.decode(z_p, K_t, pi)
        hm_p_phys = (torch.exp(hm_p * hm_log_std + hm_log_mean) - 1.0).clamp(min=0.0)
        ax.hist(hm_p_phys.cpu().numpy().flatten(), bins=60, alpha=0.6, density=True, label="N(0,1) decode")
        hm_ls, _, _ = model.inference(N_PRIOR, device, K=K_val, PI_freq=mhz, pi_freq_unit="mhz",
                                      latent_stats=latent_stats, mode="layout")
        hm_ls_phys = (torch.exp(hm_ls * hm_log_std + hm_log_mean) - 1.0).clamp(min=0.0)
        ax.hist(hm_ls_phys.cpu().numpy().flatten(), bins=60, alpha=0.6, density=True, label="layout infer")
        ax.set_title(f"K={K_val}  {mhz:.0f} MHz")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"test4_prior_sampling_{int(mhz)}MHz.png", dpi=150)
    plt.close(fig)
    print(f"  → test4_prior_sampling_{int(mhz)}MHz.png")
    return results


def test_cross_freq_same_z(model, device, data_dir, out_dir: Path, mhz_native: float = 200.0,
                           mhz_alt: float = 250.0):
    """Decode the same posterior z at two MHz — measures freq conditioning on decode."""
    print("\n" + "=" * 60)
    print(f"TEST 5 — Same-z cross-freq decode  ({mhz_native:.0f} vs {mhz_alt:.0f} MHz)")
    print("=" * 60)
    ds = VAEDataset(data_dir=str(data_dir))
    if not ds.has_pifreq:
        print("  No PI_freq in dataset; skipping.")
        return
    idx = None
    for i in range(min(len(ds), 5000)):
        s = ds[i]
        if int(s["K"]) == K_TEST_PRESENT:
            idx = i
            break
    if idx is None:
        return
    s = ds[idx]
    hm = s["heatmap_norm"].unsqueeze(0).to(device)
    occ = s["occupancy"].unsqueeze(0).to(device)
    imp = s["impedance"].unsqueeze(0).to(device)
    if imp.dim() == 2:
        imp = imp.unsqueeze(1)
    K = s["K"].unsqueeze(0).to(device)
    pi_native = torch.tensor([pi_freq_mhz_to_norm(mhz_native)], device=device)
    pi_alt = torch.tensor([pi_freq_mhz_to_norm(mhz_alt)], device=device)
    with torch.no_grad():
        _, mu, _, _ = model.encode(hm, occ, imp, K, pi_native)
        hm_n, _, _ = model.decode(mu, K, pi_native)
        hm_a, _, _ = model.decode(mu, K, pi_alt)
        hm_blend = model.decode_heatmap_blended(mu, K, mhz_alt)

    diff = (hm_a - hm_n).abs().mean().item()
    print(f"  Mean |Δheatmap| norm space (native vs alt MHz): {diff:.5f}")

    norm_stats = json.load(open(Path(data_dir) / "normalization_stats.json", encoding="utf-8"))
    hm_log_mean, hm_log_std = norm_stats["Heatmap"]["log_mean"], norm_stats["Heatmap"]["log_std"]
    to_phys = lambda t: (torch.exp(t * hm_log_std + hm_log_mean) - 1.0).clamp(min=0.0)[0, 0].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, arr, title in zip(
        axes,
        [hm_n, hm_a, hm_blend],
        [f"decode @ {mhz_native:.0f}", f"decode @ {mhz_alt:.0f}", f"blend @ {mhz_alt:.0f}"],
    ):
        ax.imshow(to_phys(arr), cmap="jet", origin="lower")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"test5_crossfreq_z_{int(mhz_native)}_{int(mhz_alt)}MHz.png", dpi=150)
    plt.close(fig)
    print(f"  → test5_crossfreq_z_{int(mhz_native)}_{int(mhz_alt)}MHz.png")


def run_off_anchor(model, cfg, device, out_dir: Path, mhz_list: tuple[float, ...], max_batches: int):
    print("\n" + "=" * 60)
    print("TEST 6 — Off-anchor val FG-MSE (encode_cross vs layout_cross)")
    print("=" * 60)
    _, val_ld = create_multifreq_data_loaders(
        data_dir=str(cfg.get("data_dir", resolve_paths(cfg)["data_dir"])),
        batch_size=int(cfg.get("batch_size", 64)),
        num_workers=2,
        train_split=float(cfg.get("train_split", 0.9)),
        seed=42,
        pin_memory=device.type == "cuda",
        split_by_design=True,
        stratify_by_k=True,
        balance_k=False,
        balance_freq=False,
        cache_in_ram=cfg.get("cache_in_ram", True),
        cross_freq_pairs=False,
    )
    csv_path = out_dir / "off_anchor_eval.csv"
    run_off_anchor_eval(
        model, val_ld,
        bg=float(cfg.get("background_value", -2.9669)),
        off_anchor_mhz=mhz_list,
        max_batches=max_batches,
        device=device,
        out_csv=csv_path,
    )


def main():
    ap = argparse.ArgumentParser(description="exp045 VAE evaluation (PI_freq)")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--mhz", nargs="*", type=float, default=list(TEST_MHZ))
    ap.add_argument("--off-anchor", nargs="*", type=float, default=list(OFF_ANCHOR_MHZ))
    ap.add_argument("--max-batches", type=int, default=30, help="Val batches for test 6 (0=all)")
    ap.add_argument("--skip-off-anchor", action="store_true")
    args = ap.parse_args()

    cfg = load_exp_config()
    paths = resolve_paths(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.ckpt) if args.ckpt else Path(paths["checkpoint"])
    out_dir = Path(args.output or (paths["exp_dir"] / "eval_results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}\nCheckpoint: {ckpt}\nOutput: {out_dir}\n")

    model, _, latent_stats, _ = load_model(ckpt, device)
    mhz_list = tuple(args.mhz)

    test_variation(model, latent_stats, device, out_dir, mhz_list)
    test_nn_distance(model, latent_stats, device, paths["data_dir"], out_dir, mhz=mhz_list[1] if len(mhz_list) > 1 else 200.0)
    for mhz in mhz_list[:2]:
        test_interpolation(model, device, paths["data_dir"], paths, mhz, out_dir)
    for mhz in mhz_list[:1]:
        test_prior_sampling(model, latent_stats, device, paths, mhz, out_dir)
    off = tuple(args.off_anchor)
    if len(off) >= 2:
        test_cross_freq_same_z(model, device, paths["data_dir"], out_dir, mhz_native=200.0, mhz_alt=off[0])
    if not args.skip_off_anchor and off:
        run_off_anchor(model, cfg, device, out_dir, off, args.max_batches)

    print("\n" + "=" * 60)
    print("Done. Results →", out_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
