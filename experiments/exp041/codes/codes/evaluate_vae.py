"""
evaluate_vae.py — Post-training evaluation for exp025_latent_size_change

Runs 4 diagnostics:
  1. Variation within K     — are 50 samples at same K actually diverse?
  2. Nearest-neighbour dist — are generated samples novel (not memorised)?
  3. Latent interpolation   — is the decoder manifold smooth?
  4. Prior sampling sanity  — does z ~ N(0,1) produce valid output?

Usage:
    python3 experiments/exp025_latent_size_change/codes/evaluate_vae.py
"""
import sys, os, json
import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = "/home/ubuntu/gan"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from experiments.exp025_latent_size_change.codes.vae_multi_input_simple import MultiInputVAE
from src_vae.others.dataloader import VAEDataset

# ─── CONFIG ──────────────────────────────────────────────────────────────────
CHECKPOINT_PATH   = "experiments/exp030_adding_physic/checkpoints/checkpoint_epoch_400.pt"
DATA_DIR          = "datasets/data_norm"
OUTPUT_DIR        = "experiments/exp030_adding_physic/eval_results"
LATENT_DIM        = 48
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test K values: pick one from the mid-range dataset (present) and one rare extreme
K_TEST_PRESENT    = 26   # common — should have good coverage
K_TEST_RARE       = 2   # rare/extreme — tests gap-filling

N_VARIATION       = 50   # samples per K for variation test
N_NN_TRAIN        = 500  # how many training samples to load for NN distance
N_PRIOR           = 200  # z ~ N(0,1) samples for prior sanity
INTERP_STEPS      = 8    # frames in interpolation
TEMPS             = [0.6, 0.8, 1.0, 1.2, 1.5]  # temperatures to sweep

FREQ_PATH         = "configs/Frequency_data_hz.npy"
TARGET_IMP_PATH   = "configs/target_impedance.npy"
BINARY_MASK_PATH  = "configs/binary_mask.npy"
NORM_STATS_PATH   = "datasets/data_norm/normalization_stats.json"
# ─────────────────────────────────────────────────────────────────────────────


def load_model(checkpoint_path: str, latent_dim: int, device: torch.device) -> tuple:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt.get("config", {})
    ld   = cfg.get("latent_dim", latent_dim)
    cond = cfg.get("cond_dim", 8)

    model = MultiInputVAE(latent_dim=ld, cond_dim=cond)
    ckpt_state  = ckpt["model_state_dict"]
    model_state = model.state_dict()
    compat = {k: v for k, v in ckpt_state.items()
              if k in model_state and model_state[k].shape == v.shape}
    model.load_state_dict(compat, strict=False)
    model.to(device).eval()

    latent_stats     = ckpt.get("latent_stats", None)
    per_K_stats      = ckpt.get("per_K_latent_stats", None)
    print(f"Loaded model from {checkpoint_path}  ({len(compat)}/{len(ckpt_state)} tensors)")
    return model, latent_stats, per_K_stats


def encode_batch(model, batch, device):
    """Run encoder on a dataset batch dict → fused mu (B, D)."""
    hm  = batch["heatmap_norm"].to(device)
    occ = batch["occupancy"].to(device)
    imp = batch["impedance"].to(device)
    K   = batch["K"].to(device)
    with torch.no_grad():
        _, mu, _, _ = model.encode(hm, occ, imp, K)
    return mu  # (B, D)


# ─── TEST 1: Variation within K ──────────────────────────────────────────────
def test_variation(model, latent_stats, device, out_dir: Path):
    print("\n" + "=" * 60)
    print("TEST 1 — Variation within K")
    print("=" * 60)

    results = {}
    for K_val in [K_TEST_PRESENT, K_TEST_RARE]:
        hm_samples = []
        occ_samples = []
        imp_samples = []
        for temp in TEMPS:
            hm_t, occ_t, imp_t = model.inference(
                N_VARIATION, device, K=K_val,
                latent_stats=latent_stats, shared_temp=temp
            )
            hm_samples.append(hm_t.cpu().numpy())    # (50, 1, 64, 64)
            occ_samples.append(torch.sigmoid(occ_t).cpu().numpy() if occ_t.dim() == 2 else occ_t.cpu().numpy())
            imp_samples.append(imp_t[:, 0].cpu().numpy())  # Ch0 only (50, 231)

        # Collect pixel stds across temps
        print(f"\n  K={K_val}:")
        row = []
        for i, temp in enumerate(TEMPS):
            hm = hm_samples[i]         # (50, 1, 64, 64)
            imp = imp_samples[i]       # (50, 231)
            hm_std  = hm.std(axis=0).mean()
            imp_std = imp.std(axis=0).mean()
            row.append((temp, hm_std, imp_std))
            print(f"    temp={temp:.1f}  hm_pixel_std={hm_std:.5f}  imp_std={imp_std:.5f}")
        results[K_val] = row

    # Plot: hm_std vs temperature for each K
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric_idx, label in zip(axes, [1, 2], ["Heatmap pixel std", "Impedance std"]):
        for K_val in [K_TEST_PRESENT, K_TEST_RARE]:
            temps_  = [r[0] for r in results[K_val]]
            vals    = [r[metric_idx] for r in results[K_val]]
            ax.plot(temps_, vals, marker="o", label=f"K={K_val}")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_xlabel("Temperature"); ax.set_ylabel(label)
        ax.set_title(f"{label} vs sampling temperature")
        ax.legend(); ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "test1_variation_vs_temperature.png", dpi=150)
    plt.close(fig)
    print(f"\n  → Saved: test1_variation_vs_temperature.png")
    print("  Interpretation:")
    print("    std ≈ 0 at temp=1.0 → mode collapse, sigma was too low")
    print("    std > 0 at temp=1.0 → healthy variation  ✓")
    return results


# ─── TEST 2: Nearest-Neighbour Distance ──────────────────────────────────────
def test_nn_distance(model, latent_stats, device, out_dir: Path):
    print("\n" + "=" * 60)
    print("TEST 2 — Nearest-Neighbour Distance (generated vs training set)")
    print("=" * 60)

    # Load a subset of training data
    ds = VAEDataset(data_dir=DATA_DIR)
    indices = np.random.choice(len(ds), min(N_NN_TRAIN, len(ds)), replace=False)

    train_hm  = []
    train_imp = []
    for idx in indices:
        s = ds[idx]
        train_hm.append(s["heatmap_norm"].float().numpy().flatten())  # type: ignore[union-attr]
        train_imp.append(s["impedance"].float().numpy()[0])           # type: ignore[union-attr]
    train_hm  = np.stack(train_hm,  axis=0)  # (N, 64*64)
    train_imp = np.stack(train_imp, axis=0)  # (N, 231)
    print(f"  Loaded {len(train_hm)} training samples for NN comparison")

    results = {}
    for K_val in [K_TEST_PRESENT, K_TEST_RARE]:
        hm_gen, _, imp_gen = model.inference(
            50, device, K=K_val, latent_stats=latent_stats, shared_temp=1.0
        )
        hm_gen  = hm_gen.cpu().numpy().reshape(50, -1)   # (50, 4096)
        imp_gen = imp_gen[:, 0].cpu().numpy()             # (50, 231)

        hm_nn_dists  = []
        imp_nn_dists = []
        for i in range(50):
            hm_d  = np.linalg.norm(train_hm  - hm_gen[i],  axis=1).min()
            imp_d = np.linalg.norm(train_imp - imp_gen[i],  axis=1).min()
            hm_nn_dists.append(hm_d)
            imp_nn_dists.append(imp_d)

        hm_nn_dists  = np.array(hm_nn_dists)
        imp_nn_dists = np.array(imp_nn_dists)
        results[K_val] = {"hm": hm_nn_dists, "imp": imp_nn_dists}

        print(f"\n  K={K_val}:")
        print(f"    Heatmap  NN dist — mean={hm_nn_dists.mean():.4f}  min={hm_nn_dists.min():.4f}  max={hm_nn_dists.max():.4f}")
        print(f"    Impedance NN dist — mean={imp_nn_dists.mean():.4f}  min={imp_nn_dists.min():.4f}  max={imp_nn_dists.max():.4f}")

    # Histogram of NN distances
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, (K_val, lbl) in enumerate([(K_TEST_PRESENT, "present"), (K_TEST_RARE, "rare")]):
        for row, key in enumerate(["hm", "imp"]):
            ax = axes[row][col]
            ax.hist(results[K_val][key], bins=20, color="steelblue", edgecolor="white", alpha=0.85)
            ax.axvline(results[K_val][key].mean(), color="red", ls="--", lw=1.5, label="mean")
            ax.set_xlabel("L2 distance to nearest training sample")
            ax.set_ylabel("Count")
            mod = "Heatmap" if row == 0 else "Impedance"
            ax.set_title(f"{mod} NN dist  K={K_val} ({lbl})")
            ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "test2_nn_distance.png", dpi=150)
    plt.close(fig)
    print(f"\n  → Saved: test2_nn_distance.png")
    print("  Interpretation:")
    print("    dist ≈ 0  → memorising training data")
    print("    dist >> 0 → novel generation (gap-filling) ✓")
    return results


# ─── TEST 3: Latent Interpolation ────────────────────────────────────────────
def test_interpolation(model, device, out_dir: Path):
    print("\n" + "=" * 60)
    print("TEST 3 — Latent Space Interpolation")
    print("=" * 60)

    ds    = VAEDataset(data_dir=DATA_DIR)
    # Pick two samples with the same K (or close)
    # scan until we get a pair with K both == K_TEST_PRESENT
    idxA, idxB = None, None
    for i in range(len(ds)):
        s = ds[i]
        if int(s["K"]) == K_TEST_PRESENT:
            if idxA is None:
                idxA = i
            elif idxB is None:
                idxB = i
                break

    if idxA is None or idxB is None:
        print(f"  Could not find 2 samples with K={K_TEST_PRESENT}, skipping.")
        return

    sA = ds[idxA]; sB = ds[idxB]

    def _batch(s):
        return {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                for k, v in s.items()}

    with torch.no_grad():
        _, muA, _, _ = model.encode(
            _batch(sA)["heatmap_norm"], _batch(sA)["occupancy"],
            _batch(sA)["impedance"],   _batch(sA)["K"]
        )
        _, muB, _, _ = model.encode(
            _batch(sB)["heatmap_norm"], _batch(sB)["occupancy"],
            _batch(sB)["impedance"],   _batch(sB)["K"]
        )

    alphas   = np.linspace(0, 1, INTERP_STEPS)
    K_tensor = torch.full((1,), K_TEST_PRESENT, dtype=torch.long, device=device)

    freq      = np.load(FREQ_PATH).squeeze()
    binary_mask = np.load(BINARY_MASK_PATH)
    norm_stats  = json.load(open(NORM_STATS_PATH))
    hm_log_mean = norm_stats["Heatmap"]["log_mean"]
    hm_log_std  = norm_stats["Heatmap"]["log_std"]
    imp_log_mean = norm_stats["Impedance"]["log_mean"]
    imp_log_std  = norm_stats["Impedance"]["log_std"]

    hm_frames  = []
    imp_frames = []
    with torch.no_grad():
        for alpha in alphas:
            z = (1.0 - alpha) * muA + alpha * muB
            hm, _, imp = model.decode(z, K_tensor)
            hm_phys = (torch.exp(hm * hm_log_std + hm_log_mean) - 1.0).clamp(min=0.0)
            hm_frames.append(hm_phys[0, 0].cpu().numpy())       # (64, 64)
            imp_frames.append(imp[0, 0].cpu().numpy())           # (231,)  Ch0

    # Plot grid: top row = heatmaps, bottom row = impedance
    ncols = INTERP_STEPS
    fig   = plt.figure(figsize=(ncols * 3, 7))
    gs    = gridspec.GridSpec(2, ncols, hspace=0.35, wspace=0.1)

    cmap_hm = plt.get_cmap("jet")
    vmin = min(f.min() for f in hm_frames)
    vmax = max(f.max() for f in hm_frames)
    target_imp = np.load(TARGET_IMP_PATH).squeeze()

    for j, alpha in enumerate(alphas):
        ax_hm = fig.add_subplot(gs[0, j])
        hm_masked = np.ma.masked_where(~binary_mask, hm_frames[j])
        ax_hm.imshow(hm_masked, cmap=cmap_hm, vmin=vmin, vmax=vmax,
                     aspect="auto", origin="lower", interpolation="bicubic")
        ax_hm.set_title(f"α={alpha:.2f}", fontsize=8)
        ax_hm.axis("off")

        ax_imp = fig.add_subplot(gs[1, j])
        ax_imp.loglog(freq, target_imp, "--", lw=1, color="red", alpha=0.5)
        ax_imp.loglog(freq, np.exp(imp_frames[j] * imp_log_std + imp_log_mean),
                      "-", lw=1.5, color="royalblue")
        ax_imp.set_ylim(1e-3, 1e2)
        ax_imp.set_xticks([]); ax_imp.set_yticks([])
        ax_imp.grid(True, which="both", alpha=0.2)

    fig.suptitle(f"Latent interpolation: A → B  (K={K_TEST_PRESENT})", fontsize=13, fontweight="bold")
    fig.savefig(out_dir / "test3_interpolation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Interpolated between sample {idxA} and {idxB}")
    print(f"  → Saved: test3_interpolation.png")
    print("  Interpretation:")
    print("    Smooth gradual change → healthy manifold  ✓")
    print("    Abrupt jump / garbage midpoints → holes in latent space  ✗")


# ─── TEST 4: Prior Sampling Sanity ───────────────────────────────────────────
def test_prior_sampling(model, latent_stats, device, out_dir: Path):
    print("\n" + "=" * 60)
    print("TEST 4 — Prior Sampling Sanity  (z ~ N(0,1))")
    print("=" * 60)

    norm_stats  = json.load(open(NORM_STATS_PATH))
    hm_log_mean = norm_stats["Heatmap"]["log_mean"]
    hm_log_std  = norm_stats["Heatmap"]["log_std"]
    imp_log_mean = norm_stats["Impedance"]["log_mean"]
    imp_log_std  = norm_stats["Impedance"]["log_std"]

    results = {}
    for K_val in [K_TEST_PRESENT, K_TEST_RARE]:
        # A) Pure N(0,1) — no latent stats correction
        z_prior = torch.randn(N_PRIOR, model.latent_dim, device=device)
        K_tensor = torch.full((N_PRIOR,), K_val, dtype=torch.long, device=device)
        with torch.no_grad():
            hm, occ, imp = model.decode(z_prior, K_tensor)
            occ_prob = torch.sigmoid(occ)

        hm_phys = (torch.exp(hm * hm_log_std + hm_log_mean) - 1.0).clamp(min=0.0)
        hm_np   = hm_phys[:, 0].cpu().numpy()   # (N, 64, 64)
        imp_np  = imp[:, 0].cpu().numpy()        # (N, 231)  Ch0

        hm_mean  = hm_np.mean()
        hm_std   = hm_np.std()
        imp_mean = imp_np.mean()
        imp_std  = imp_np.std()
        occ_mean = occ_prob.mean().item()

        # Degenerate check: std ≈ 0 means decoder ignores z
        degenerate = hm_std < 0.001

        # B) Using latent stats (correct generation)
        hm_ls, _, imp_ls = model.inference(
            N_PRIOR, device, K=K_val, latent_stats=latent_stats, shared_temp=1.0
        )
        hm_ls_phys = (torch.exp(hm_ls * hm_log_std + hm_log_mean) - 1.0).clamp(min=0.0)
        hm_ls_std  = hm_ls_phys[:, 0].cpu().numpy().std()

        results[K_val] = {
            "prior_hm_mean":  hm_mean,
            "prior_hm_std":   hm_std,
            "prior_imp_std":  imp_std,
            "prior_occ_mean": occ_mean,
            "latent_stats_hm_std": hm_ls_std,
            "degenerate":     degenerate,
        }

        status = "❌ DEGENERATE (decoder ignores z — sigma likely too low)" if degenerate else "✓ OK"
        print(f"\n  K={K_val}:")
        print(f"    Pure N(0,1):   hm_mean={hm_mean:.4f}  hm_std={hm_std:.5f}  imp_std={imp_std:.5f}  {status}")
        print(f"    With lat.stats: hm_std={hm_ls_std:.5f}")
        print(f"    occ_prob_mean={occ_mean:.4f}  (expected ~{K_val/52:.2f})")

    # Plot: distribution of heatmap pixel values for pure prior vs latent-stats
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, K_val in zip(axes, [K_TEST_PRESENT, K_TEST_RARE]):
        # Pure prior
        z_p = torch.randn(N_PRIOR, model.latent_dim, device=device)
        K_t = torch.full((N_PRIOR,), K_val, dtype=torch.long, device=device)
        with torch.no_grad():
            hm_p, _, _ = model.decode(z_p, K_t)
        hm_p_phys = (torch.exp(hm_p * hm_log_std + hm_log_mean) - 1.0).clamp(min=0.0)
        ax.hist(hm_p_phys.cpu().numpy().flatten(), bins=60, alpha=0.6,
                color="tomato", label="pure N(0,1)", density=True)

        # With latent stats
        hm_ls2, _, _ = model.inference(N_PRIOR, device, K=K_val, latent_stats=latent_stats)
        hm_ls2_phys = (torch.exp(hm_ls2 * hm_log_std + hm_log_mean) - 1.0).clamp(min=0.0)
        ax.hist(hm_ls2_phys.cpu().numpy().flatten(), bins=60, alpha=0.6,
                color="steelblue", label="with latent stats", density=True)

        ax.set_xlabel("Heatmap physical value"); ax.set_ylabel("Density")
        ax.set_title(f"Prior sampling distribution  K={K_val}")
        ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

    fig.tight_layout()
    fig.savefig(out_dir / "test4_prior_sampling.png", dpi=150)
    plt.close(fig)
    print(f"\n  → Saved: test4_prior_sampling.png")
    print("  Interpretation:")
    print("    Pure prior std >> 0      → sigma_reg has effect, decoder uses z  ✓")
    print("    Latent-stats std > Prior → latent stats are helping diversity  ✓")
    return results


# ─── SUMMARY ─────────────────────────────────────────────────────────────────
def print_summary(var_results, nn_results, prior_results):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for K_val in [K_TEST_PRESENT, K_TEST_RARE]:
        label = "present in dataset" if K_val == K_TEST_PRESENT else "rare/extreme"
        print(f"\n  K={K_val}  ({label})")

        # Variation at temp=1.0
        var_row = next(r for r in var_results[K_val] if r[0] == 1.0)
        hm_std = var_row[1]
        if hm_std < 0.001:
            var_verdict = "❌ mode collapse — all samples identical"
        elif hm_std < 0.01:
            var_verdict = "⚠  low diversity — minor variation only"
        else:
            var_verdict = "✓  healthy variation"
        print(f"    Variation @temp=1.0 : hm_std={hm_std:.5f}  → {var_verdict}")

        # NN distance
        nn_hm = nn_results[K_val]["hm"].mean()
        if nn_hm < 0.5:
            nn_verdict = "⚠  possibly memorising training data"
        else:
            nn_verdict = "✓  generating novel (gap-filling) samples"
        print(f"    NN dist (heatmap)   : mean={nn_hm:.4f}  → {nn_verdict}")

        # Prior sanity
        p = prior_results[K_val]
        if p["degenerate"]:
            prior_verdict = "❌ decoder ignores z (sigma too low)"
        else:
            prior_verdict = "✓  decoder responds to z variation"
        print(f"    Prior sampling      : hm_std={p['prior_hm_std']:.5f}  → {prior_verdict}")

    print("\n  See eval_results/ for all plots.")


# ─── SCORECARD ───────────────────────────────────────────────────────────────
def plot_model_scorecard(var_results, nn_results, prior_results, out_dir: Path):
    """
    Single-page model health scorecard.

    Layout (2 rows × 3 cols):
      [0,0] Pass/fail checklist     [0,1] Diversity bar chart   [0,2] NN distance bars
      [1,0] Overall verdict banner  [1,1] Prior σ response      [1,2] Temp sensitivity
    """
    import matplotlib.patches as mpatches

    PASS_COLOR = "#2ecc71"    # green
    WARN_COLOR = "#f39c12"    # amber
    FAIL_COLOR = "#e74c3c"    # red
    BG_COLOR   = "#1e1e2e"
    FG_COLOR   = "#f0f0f0"

    fig = plt.figure(figsize=(18, 10), facecolor=BG_COLOR)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                            left=0.06, right=0.97, top=0.93, bottom=0.08)

    # ── collect metrics ──────────────────────────────────────────────────────
    checks = []   # (label, value_str, verdict)  verdict ∈ {"pass","warn","fail"}
    score  = 0
    total  = 0

    for K_val, label in [(K_TEST_PRESENT, "present"), (K_TEST_RARE, "rare")]:
        # Variation
        var_row  = next(r for r in var_results[K_val] if r[0] == 1.0)
        hm_std   = var_row[1]
        imp_std  = var_row[2]
        if hm_std > 0.01:
            v = "pass"; score += 1
        elif hm_std > 0.001:
            v = "warn"
        else:
            v = "fail"
        checks.append((f"Diversity  K={K_val} ({label})", f"hm_std={hm_std:.4f}", v))
        total += 1

        # NN distance
        nn_mean = nn_results[K_val]["hm"].mean()
        if nn_mean > 1.0:
            v = "pass"; score += 1
        elif nn_mean > 0.5:
            v = "warn"; score += 0.5
        else:
            v = "fail"
        checks.append((f"Novelty    K={K_val} ({label})", f"NN={nn_mean:.3f}", v))
        total += 1

        # Prior response
        p = prior_results[K_val]
        if not p["degenerate"] and p["prior_hm_std"] > 0.005:
            v = "pass"; score += 1
        elif not p["degenerate"]:
            v = "warn"; score += 0.5
        else:
            v = "fail"
        checks.append((f"Prior resp K={K_val} ({label})", f"hm_std={p['prior_hm_std']:.5f}", v))
        total += 1

    score_pct = score / total

    # ── panel [0,0]: pass/fail checklist ────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_facecolor(BG_COLOR)
    ax0.set_xlim(0, 1); ax0.set_ylim(0, len(checks) + 0.5)
    ax0.axis("off")
    ax0.set_title("Diagnostic Checks", color=FG_COLOR, fontsize=11, fontweight="bold", pad=6)

    color_map = {"pass": PASS_COLOR, "warn": WARN_COLOR, "fail": FAIL_COLOR}
    icon_map  = {"pass": "●  PASS", "warn": "◑  WARN", "fail": "○  FAIL"}

    for i, (lbl, val, verdict) in enumerate(reversed(checks)):
        y = i + 0.6
        c = color_map[verdict]
        ax0.add_patch(mpatches.FancyBboxPatch(
            (0.02, y - 0.35), 0.96, 0.7,
            boxstyle="round,pad=0.02", facecolor=c + "30", edgecolor=c, lw=1.2))
        ax0.text(0.07, y, f"{icon_map[verdict]}  {lbl}", color=c,
                 fontsize=7.5, va="center")
        ax0.text(0.93, y, val, color=FG_COLOR, fontsize=7, va="center", ha="right")

    # ── panel [0,1]: diversity bar chart ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_facecolor(BG_COLOR)
    ax1.tick_params(colors=FG_COLOR)
    for sp in ax1.spines.values():
        sp.set_edgecolor("#444")

    bar_labels = [f"K={k}\n({lbl})" for k, lbl in
                  [(K_TEST_PRESENT, "present"), (K_TEST_RARE, "rare")]]
    hm_stds  = [next(r for r in var_results[k] if r[0] == 1.0)[1]
                for k in [K_TEST_PRESENT, K_TEST_RARE]]
    imp_stds = [next(r for r in var_results[k] if r[0] == 1.0)[2]
                for k in [K_TEST_PRESENT, K_TEST_RARE]]

    x = np.arange(2)
    ax1.bar(x - 0.2, hm_stds,  0.35, label="Heatmap std",   color="#3498db", alpha=0.85)
    ax1.bar(x + 0.2, imp_stds, 0.35, label="Impedance std", color="#e67e22", alpha=0.85)
    ax1.axhline(0.01,  color=PASS_COLOR, ls="--", lw=1.2, alpha=0.7, label="Good (0.01)")
    ax1.axhline(0.001, color=FAIL_COLOR, ls=":",  lw=1.2, alpha=0.7, label="Collapse (0.001)")
    ax1.set_xticks(x); ax1.set_xticklabels(bar_labels, color=FG_COLOR, fontsize=9)
    ax1.set_ylabel("Std @ temp=1.0", color=FG_COLOR, fontsize=9)
    ax1.set_title("Sample Diversity (temp=1.0)", color=FG_COLOR, fontsize=11, fontweight="bold")
    ax1.legend(fontsize=7, framealpha=0.3, labelcolor=FG_COLOR)
    ax1.yaxis.label.set_color(FG_COLOR)
    ax1.set_facecolor(BG_COLOR)
    ax1.title.set_color(FG_COLOR)

    # ── panel [0,2]: NN distance bars ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(BG_COLOR)
    ax2.tick_params(colors=FG_COLOR)
    for sp in ax2.spines.values():
        sp.set_edgecolor("#444")

    nn_hm  = [nn_results[k]["hm"].mean()  for k in [K_TEST_PRESENT, K_TEST_RARE]]
    nn_imp = [nn_results[k]["imp"].mean() for k in [K_TEST_PRESENT, K_TEST_RARE]]
    ax2.bar(x - 0.2, nn_hm,  0.35, label="Heatmap",   color="#9b59b6", alpha=0.85)
    ax2.bar(x + 0.2, nn_imp, 0.35, label="Impedance", color="#1abc9c", alpha=0.85)
    ax2.axhline(1.0, color=PASS_COLOR, ls="--", lw=1.2, alpha=0.7, label="Novel (1.0)")
    ax2.axhline(0.5, color=WARN_COLOR, ls=":",  lw=1.2, alpha=0.7, label="Borderline (0.5)")
    ax2.set_xticks(x); ax2.set_xticklabels(bar_labels, color=FG_COLOR, fontsize=9)
    ax2.set_ylabel("Mean L2 NN distance", color=FG_COLOR, fontsize=9)
    ax2.set_title("Novelty (↑ = not memorising)", color=FG_COLOR, fontsize=11, fontweight="bold")
    ax2.legend(fontsize=7, framealpha=0.3, labelcolor=FG_COLOR)
    ax2.yaxis.label.set_color(FG_COLOR)

    # ── panel [1,0]: overall verdict ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(BG_COLOR); ax3.axis("off")
    if score_pct >= 0.8:
        verdict_text  = "GOOD"
        verdict_color = PASS_COLOR
        sub_text      = "Model encodes real diversity\nand generates novel samples."
    elif score_pct >= 0.5:
        verdict_text  = "MARGINAL"
        verdict_color = WARN_COLOR
        sub_text      = "Partial diversity. Some collapse\nor memorisation detected."
    else:
        verdict_text  = "POOR"
        verdict_color = FAIL_COLOR
        sub_text      = "Significant collapse or\nmemorisation — needs retraining."

    ax3.text(0.5, 0.68, verdict_text, ha="center", va="center",
             fontsize=42, fontweight="bold", color=verdict_color,
             transform=ax3.transAxes)
    ax3.text(0.5, 0.4, sub_text, ha="center", va="center",
             fontsize=10, color=FG_COLOR, transform=ax3.transAxes)
    ax3.text(0.5, 0.18, f"Score: {score:.1f} / {total}  ({score_pct*100:.0f}%)",
             ha="center", va="center", fontsize=12, color=FG_COLOR,
             transform=ax3.transAxes)
    ax3.set_title("Overall Verdict", color=FG_COLOR, fontsize=11, fontweight="bold", pad=6)

    # ── panel [1,1]: prior response — box plots ──────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(BG_COLOR)
    ax4.tick_params(colors=FG_COLOR)
    for sp in ax4.spines.values():
        sp.set_edgecolor("#444")

    prior_stds   = [prior_results[k]["prior_hm_std"]         for k in [K_TEST_PRESENT, K_TEST_RARE]]
    latent_stds  = [prior_results[k]["latent_stats_hm_std"]   for k in [K_TEST_PRESENT, K_TEST_RARE]]
    ax4.bar(x - 0.2, prior_stds,  0.35, label="Pure N(0,1)",   color="#e74c3c", alpha=0.85)
    ax4.bar(x + 0.2, latent_stds, 0.35, label="Latent stats",  color="#2980b9", alpha=0.85)
    ax4.axhline(0.001, color=FAIL_COLOR, ls=":", lw=1.2, alpha=0.7, label="Collapse threshold")
    ax4.set_xticks(x); ax4.set_xticklabels(bar_labels, color=FG_COLOR, fontsize=9)
    ax4.set_ylabel("Heatmap output std", color=FG_COLOR, fontsize=9)
    ax4.set_title("Prior Sampling Response", color=FG_COLOR, fontsize=11, fontweight="bold")
    ax4.legend(fontsize=7, framealpha=0.3, labelcolor=FG_COLOR)
    ax4.yaxis.label.set_color(FG_COLOR)

    # ── panel [1,2]: diversity vs temperature ───────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(BG_COLOR)
    ax5.tick_params(colors=FG_COLOR)
    for sp in ax5.spines.values():
        sp.set_edgecolor("#444")

    colors_k = ["#3498db", "#e67e22"]
    for k_idx, K_val in enumerate([K_TEST_PRESENT, K_TEST_RARE]):
        temps_  = [r[0] for r in var_results[K_val]]
        hm_s    = [r[1] for r in var_results[K_val]]
        lbl_str = "present" if K_val == K_TEST_PRESENT else "rare"
        ax5.plot(temps_, hm_s, marker="o", color=colors_k[k_idx],
                 lw=2, label=f"K={K_val} ({lbl_str})")
    ax5.axhline(0.01,  color=PASS_COLOR, ls="--", lw=1.2, alpha=0.7)
    ax5.axhline(0.001, color=FAIL_COLOR, ls=":",  lw=1.2, alpha=0.7)
    ax5.axvline(1.0, color="white", ls=":", lw=0.8, alpha=0.4)
    ax5.set_xlabel("Temperature", color=FG_COLOR, fontsize=9)
    ax5.set_ylabel("Heatmap std", color=FG_COLOR, fontsize=9)
    ax5.set_title("Diversity vs Temperature", color=FG_COLOR, fontsize=11, fontweight="bold")
    ax5.legend(fontsize=8, framealpha=0.3, labelcolor=FG_COLOR)
    ax5.xaxis.label.set_color(FG_COLOR)
    ax5.yaxis.label.set_color(FG_COLOR)

    fig.suptitle("Model Health Scorecard — exp025_latent_size_change",
                 color=FG_COLOR, fontsize=14, fontweight="bold", y=0.98)

    out_path = out_dir / "scorecard.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"\n  → Saved: scorecard.png  (verdict: {verdict_text}  {score:.1f}/{total})")


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Output: {out_dir}")

    model, latent_stats, per_K_stats = load_model(CHECKPOINT_PATH, LATENT_DIM, DEVICE)
    model.eval()

    var_results   = test_variation(model, latent_stats, DEVICE, out_dir)
    nn_results    = test_nn_distance(model, latent_stats, DEVICE, out_dir)
    test_interpolation(model, DEVICE, out_dir)
    prior_results = test_prior_sampling(model, latent_stats, DEVICE, out_dir)

    print_summary(var_results, nn_results, prior_results)
    plot_model_scorecard(var_results, nn_results, prior_results, out_dir)


if __name__ == "__main__":
    main()
