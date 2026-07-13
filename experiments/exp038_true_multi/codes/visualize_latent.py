"""
visualize_latent.py — Latent space visualizations for exp025_latent_size_change

Encodes the full dataset → collects all fused mu vectors → produces:

  Plot 1 — t-SNE 2D scatter    colored by K
  Plot 2 — PCA  2D scatter     colored by K
  Plot 3 — PCA explained variance  (how many dims the model uses)
  Plot 4 — Per-dim mu / sigma  (detect posterior collapse)
  Plot 5 — 2D decoder grid     (traverse top-2 PCA axes, decode at each grid point)
  Plot 6 — Fused sigma histogram per K bucket  (is sigma_reg working?)
    Plot 7 — KL per latent dim   (information usage per dim)
    Plot 8 — KL distribution per K bucket
    Plot 9 — Linear probe: predict K from mu (R²)
    Plot 10 — Modality-vs-fused disagreement vs K (experts vs PoE fused)

Usage:
    cd /home/ubuntu/genai_pdn
    python3 experiments/exp030_adding_physic/codes/visualize_latent.py
"""
import sys, json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from repo_paths import REPO_ROOT as PROJECT_ROOT, setup_path
setup_path()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from experiments.exp025_latent_size_change.codes.vae_multi_input_simple import MultiInputVAE
from src_vae.others.dataloader import VAEDataset

# ─── CONFIG ──────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "experiments/exp030_adding_physic/checkpoints/checkpoint_epoch_400.pt"
DATA_DIR        = "datasets/data_norm"
OUTPUT_DIR      = "experiments/exp030_adding_physic/latent_visuals_1"
LATENT_DIM      = 32
BATCH_SIZE      = 128
MAX_SAMPLES     = 30000   # cap for t-SNE speed (t-SNE is O(n²))
TSNE_PERPLEXITY = 40

NORM_STATS_PATH = "datasets/data_norm/normalization_stats.json"
BINARY_MASK_PATH = "configs/binary_mask.npy"
FREQ_PATH        = "configs/Frequency_data_hz.npy"
TARGET_IMP_PATH  = "configs/target_impedance.npy"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ─────────────────────────────────────────────────────────────────────────────


# ── helpers ──────────────────────────────────────────────────────────────────
def load_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    cfg  = ckpt.get("config", {})
    ld   = cfg.get("latent_dim", LATENT_DIM)
    cond = cfg.get("cond_dim", 8)
    model = MultiInputVAE(latent_dim=ld, cond_dim=cond)
    state = ckpt["model_state_dict"]
    ms    = model.state_dict()
    compat = {k: v for k, v in state.items() if k in ms and ms[k].shape == v.shape}
    model.load_state_dict(compat, strict=False)
    model.to(DEVICE).eval()
    print(f"Loaded model  latent_dim={ld}  ({len(compat)}/{len(state)} tensors)")
    return model


def encode_dataset(model, data_dir, max_samples=MAX_SAMPLES, collect_experts=True):
    """Encode dataset subset → return fused stats + optional per-modality expert stats."""
    ds = VAEDataset(data_dir=data_dir)
    n  = min(len(ds), max_samples)
    idx = np.random.choice(len(ds), n, replace=False).tolist()
    sub = torch.utils.data.Subset(ds, idx)
    dl  = DataLoader(sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    all_mu     = []
    all_logvar = []
    all_K      = []

    expert_mu = {"heatmap": [], "occupancy": [], "impedance": []}
    expert_lv = {"heatmap": [], "occupancy": [], "impedance": []}

    with torch.no_grad():
        for batch in dl:
            hm  = batch["heatmap_norm"].to(DEVICE)
            occ = batch["occupancy"].to(DEVICE)
            imp = batch["impedance"].to(DEVICE)
            K   = batch["K"].to(DEVICE)
            _, mu, logvar, expert_stats = model.encode(hm, occ, imp, K)
            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())
            all_K.append(K.cpu().numpy())

            if collect_experts:
                for m in ("heatmap", "occupancy", "impedance"):
                    m_mu, m_lv = expert_stats[m]
                    expert_mu[m].append(m_mu.cpu().numpy())
                    expert_lv[m].append(m_lv.cpu().numpy())

    mu     = np.concatenate(all_mu,     axis=0)   # (N, D)
    logvar = np.concatenate(all_logvar, axis=0)   # (N, D)
    K_arr  = np.concatenate(all_K,      axis=0)   # (N,)
    sigma  = np.exp(0.5 * logvar)                  # (N, D)
    print(f"Encoded {len(mu)} samples  (out of {len(ds)} total)")

    experts = None
    if collect_experts:
        experts = {}
        for m in ("heatmap", "occupancy", "impedance"):
            m_mu = np.concatenate(expert_mu[m], axis=0)
            m_lv = np.concatenate(expert_lv[m], axis=0)
            experts[m] = {
                "mu": m_mu,
                "logvar": m_lv,
                "sigma": np.exp(0.5 * m_lv),
            }
    return mu, logvar, sigma, K_arr, experts


def kl_diag_standard_normal(mu, logvar):
    """KL(q(z|x)=N(mu,diag(exp(logvar))) || p(z)=N(0,I)). Returns (N,D)."""
    var = np.exp(logvar)
    return 0.5 * (mu ** 2 + var - 1.0 - logvar)


def _bucket_specs():
    return [(0, 13, "K 0–12"), (13, 26, "K 13–25"), (26, 39, "K 26–38"), (39, 53, "K 39–52")]


def k_cmap(K_arr):
    """Return RGBA colors mapped from K values (0–52)."""
    norm  = Normalize(vmin=0, vmax=52)
    cmap  = plt.get_cmap("plasma")
    return cmap(norm(K_arr)), norm, cmap


# ── Plot 1 — t-SNE ───────────────────────────────────────────────────────────
def plot_tsne(mu, K_arr, out_dir):
    print("\nComputing t-SNE (may take ~1–2 min)…")
    tsne  = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, random_state=42, max_iter=1000)
    emb   = tsne.fit_transform(mu)          # (N, 2)

    colors, norm, cmap = k_cmap(K_arr)
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=K_arr, cmap="plasma",
                    norm=norm, s=8, alpha=0.65, linewidths=0)
    plt.colorbar(sc, ax=ax, label="K (occupied slots)")
    ax.set_title("t-SNE of fused posterior means  (colored by K)", fontsize=13)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "plot1_tsne_by_K.png", dpi=150)
    plt.close(fig)
    print("  → plot1_tsne_by_K.png")
    return emb


# ── Plot 2 — PCA scatter ─────────────────────────────────────────────────────
def plot_pca_scatter(mu, K_arr, out_dir):
    pca = PCA(n_components=10, random_state=42)
    pca.fit(mu)
    emb2 = pca.transform(mu)[:, :2]        # top-2 PCs only

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(emb2[:, 0], emb2[:, 1], c=K_arr, cmap="plasma",
                    norm=Normalize(vmin=0, vmax=52), s=8, alpha=0.65, linewidths=0)
    plt.colorbar(sc, ax=ax, label="K (occupied slots)")
    ax.set_title("PCA of fused posterior means  (top-2 PCs, colored by K)", fontsize=13)
    ax.set_xlabel(f"PC1  ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2  ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "plot2_pca_scatter.png", dpi=150)
    plt.close(fig)
    print("  → plot2_pca_scatter.png")
    return pca, emb2


# ── Plot 3 — PCA explained variance ─────────────────────────────────────────
def plot_pca_variance(mu, out_dir):
    pca_full = PCA(n_components=min(mu.shape[1], mu.shape[0]), random_state=42)
    pca_full.fit(mu)
    ev   = pca_full.explained_variance_ratio_
    cumev = np.cumsum(ev)

    # How many dims to reach 90%, 95%, 99%
    dims_90 = int(np.searchsorted(cumev, 0.90)) + 1
    dims_95 = int(np.searchsorted(cumev, 0.95)) + 1
    dims_99 = int(np.searchsorted(cumev, 0.99)) + 1
    print(f"\n  PCA — dims to explain 90%={dims_90}  95%={dims_95}  99%={dims_99}  (of {mu.shape[1]} total)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.bar(range(1, min(31, len(ev)+1)), ev[:30] * 100, color="steelblue", alpha=0.8)
    ax1.set_xlabel("Principal component"); ax1.set_ylabel("Explained variance (%)")
    ax1.set_title("Per-component variance (top 30)")
    ax1.grid(True, axis="y", alpha=0.35)

    ax2.plot(range(1, len(cumev)+1), cumev * 100, color="steelblue", lw=2)
    for thr, d, col in [(90, dims_90, "gold"), (95, dims_95, "orange"), (99, dims_99, "tomato")]:
        ax2.axhline(thr, ls="--", lw=1, color=col, alpha=0.7)
        ax2.axvline(d,   ls="--", lw=1, color=col, alpha=0.7, label=f"{thr}% @ dim {d}")
    ax2.set_xlabel("Number of components"); ax2.set_ylabel("Cumulative variance (%)")
    ax2.set_title("Cumulative explained variance")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.35)
    ax2.set_xlim(0, min(mu.shape[1], 60))

    fig.suptitle("Latent space dimensionality usage", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "plot3_pca_variance.png", dpi=150)
    plt.close(fig)
    print("  → plot3_pca_variance.png")
    print("  Interpretation:")
    print("    Few dims explain most variance → many latent dims unused (dead/wasted)")
    print("    Spread across many dims       → latent space fully utilised  ✓")
    return pca_full


# ── Plot 4 — Per-dim mu and sigma ────────────────────────────────────────────
def plot_per_dim(mu, sigma, out_dir):
    dim_mu_mean  = mu.mean(axis=0)     # (D,)
    dim_mu_std   = mu.std(axis=0)      # (D,)  — spread across dataset
    dim_sigma_mean = sigma.mean(axis=0) # (D,)  — avg posterior uncertainty per dim

    # Sort dims by mu_std descending (most-active dims first)
    order = np.argsort(dim_mu_std)[::-1]
    D = mu.shape[1]
    x = np.arange(D)

    # Count "dead" dims: where sigma_mean ≈ 1 and mu_std ≈ 0
    dead_threshold_sigma = 0.95
    dead_threshold_mu    = 0.05
    dead_dims = int(((dim_sigma_mean > dead_threshold_sigma) & (dim_mu_std < dead_threshold_mu)).sum())
    print(f"\n  Per-dim analysis:")
    print(f"    Dead dims (sigma>0.95, mu_std<0.05): {dead_dims}/{D}")
    print(f"    Mean fused sigma: {dim_sigma_mean.mean():.4f}  (target=0.45)")
    print(f"    Min fused sigma:  {dim_sigma_mean.min():.4f}")
    print(f"    Max fused sigma:  {dim_sigma_mean.max():.4f}")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 11), sharex=False)

    # Top row: mu_std per dim (sorted) — bars
    ax1.bar(x, dim_mu_std[order], color="#4C72B0", alpha=0.8, width=1.0)
    ax1.axhline(dead_threshold_mu, ls="--", color="tomato", lw=1, label=f"dead threshold ({dead_threshold_mu})")
    ax1.set_ylabel("μ std across dataset"); ax1.set_title("Per-dim posterior mean spread  (sorted)")
    ax1.set_xlabel("Latent dim (sorted by activity)"); ax1.legend(fontsize=8)
    ax1.grid(True, axis="y", alpha=0.3)

    # Middle: sigma_mean per dim (sorted same order)
    ax2.bar(x, dim_sigma_mean[order], color="#55A868", alpha=0.8, width=1.0)
    ax2.axhline(1.0, ls="--", color="grey", lw=1, label="prior σ=1")
    ax2.axhline(0.45, ls="--", color="orange", lw=1.5, label="target σ=0.45")
    ax2.axhline(dead_threshold_sigma, ls=":", color="tomato", lw=1, label=f"dead threshold ({dead_threshold_sigma})")
    ax2.set_ylabel("Mean posterior σ"); ax2.set_title("Per-dim mean fused sigma  (same sort order)")
    ax2.set_xlabel("Latent dim"); ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)

    # Bottom: 2D heatmap — mu per sample (subset 200) sorted by K
    sub_n   = min(200, mu.shape[0])
    sub_idx = np.random.choice(mu.shape[0], sub_n, replace=False)
    sub_mu  = mu[sub_idx]                      # (sub_n, D)
    sub_K   = np.zeros(sub_n, dtype=int)       # placeholder, filled below via random access
    # We can't re-look up K easily here, just show the mu heatmap sorted by PCA order
    im = ax3.imshow(sub_mu[:, order].T, aspect="auto", cmap="RdBu_r",
                    vmin=-2.5, vmax=2.5, interpolation="nearest")
    plt.colorbar(im, ax=ax3, fraction=0.015, label="μ value")
    ax3.set_xlabel("Sample index"); ax3.set_ylabel("Latent dim (sorted by activity)")
    ax3.set_title("Per-sample posterior μ  (200 random samples, dims sorted by activity)")

    fig.tight_layout(h_pad=2.5)
    fig.savefig(out_dir / "plot4_per_dim_mu_sigma.png", dpi=150)
    plt.close(fig)
    print(f"  → plot4_per_dim_mu_sigma.png")
    print(f"  Dead dims: {dead_dims}/{D}  → {'⚠ many wasted dims' if dead_dims > D//3 else '✓ most dims active'}")


# ── Plot 5 — 2D decoder grid (PC1 × PC2) ─────────────────────────────────────
def plot_decoder_grid(model, pca, out_dir, K_val=26, grid_n=8):
    """Traverse the top-2 PCA axes and decode at each grid point."""
    print(f"\n  Generating decoder grid (K={K_val}, {grid_n}×{grid_n})…")

    norm_stats  = json.load(open(NORM_STATS_PATH))
    hm_log_mean = norm_stats["Heatmap"]["log_mean"]
    hm_log_std  = norm_stats["Heatmap"]["log_std"]
    binary_mask = np.load(BINARY_MASK_PATH)

    # PC1 and PC2 axes in latent space
    pc1 = torch.tensor(pca.components_[0], dtype=torch.float32, device=DEVICE)  # (D,)
    pc2 = torch.tensor(pca.components_[1], dtype=torch.float32, device=DEVICE)  # (D,)
    origin = torch.tensor(pca.mean_, dtype=torch.float32, device=DEVICE)        # (D,)

    # Range: ±3 std of each PC
    pc1_std = float(np.sqrt(pca.explained_variance_[0]))
    pc2_std = float(np.sqrt(pca.explained_variance_[1]))
    rng1 = np.linspace(-3 * pc1_std, 3 * pc1_std, grid_n)
    rng2 = np.linspace(-3 * pc2_std, 3 * pc2_std, grid_n)[::-1]  # y-axis: top=positive

    K_tensor = torch.full((1,), K_val, dtype=torch.long, device=DEVICE)

    cmap_hm = plt.get_cmap("jet")
    fig, axes = plt.subplots(grid_n, grid_n, figsize=(grid_n * 2.2, grid_n * 2.2))
    fig.suptitle(f"Decoder grid: PC1 (→) × PC2 (↑)  K={K_val}", fontsize=14, fontweight="bold")

    with torch.no_grad():
        for row, a2 in enumerate(rng2):
            for col, a1 in enumerate(rng1):
                z = origin + a1 * pc1 + a2 * pc2   # (D,)
                z = z.unsqueeze(0)                   # (1, D)
                hm, _, _ = model.decode(z, K_tensor)
                hm_phys = (torch.exp(hm * hm_log_std + hm_log_mean) - 1.0).clamp(min=0.0)
                hm_np   = hm_phys[0, 0].cpu().numpy()   # (64, 64)
                hm_masked = np.ma.masked_where(~binary_mask, hm_np)

                ax = axes[row][col]
                ax.imshow(hm_masked, cmap=cmap_hm, aspect="auto",
                          origin="lower", interpolation="bicubic")
                ax.axis("off")
                if row == grid_n - 1:
                    ax.set_xlabel(f"{a1:.1f}", fontsize=6)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02, wspace=0.05, hspace=0.05)
    fig.savefig(out_dir / f"plot5_decoder_grid_K{K_val}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → plot5_decoder_grid_K{K_val}.png")
    print("  Interpretation:")
    print("    Smooth gradual changes across grid → manifold is continuous  ✓")
    print("    Abrupt holes / garbled patches     → gaps in learned space  ✗")


# ── Plot 6 — Fused sigma histogram by K bucket ───────────────────────────────
def plot_sigma_by_K(sigma, K_arr, out_dir):
    """Show distribution of per-sample mean sigma, bucketed by K range."""
    sigma_mean_per_sample = sigma.mean(axis=1)   # (N,)
    buckets = [(0, 13, "K 0–12"), (13, 26, "K 13–25"),
               (26, 39, "K 26–38"), (39, 53, "K 39–52")]

    fig, axes = plt.subplots(1, len(buckets), figsize=(14, 4), sharey=True)
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    overall_mean = sigma_mean_per_sample.mean()
    print(f"\n  Fused sigma statistics:")
    for ax, (lo, hi, label), col in zip(axes, buckets, colors):
        mask = (K_arr >= lo) & (K_arr < hi)
        vals = sigma_mean_per_sample[mask]
        if len(vals) == 0:
            ax.set_title(f"{label}\n(no samples)")
            continue
        ax.hist(vals, bins=30, color=col, alpha=0.8, edgecolor="white", density=True)
        ax.axvline(vals.mean(),  color="black", ls="-",  lw=1.5, label=f"mean={vals.mean():.3f}")
        ax.axvline(0.45,         color="orange", ls="--",lw=1.5, label="target=0.45")
        ax.axvline(0.5,          color="red",    ls=":",  lw=1,   label="ceil≈0.5")
        ax.set_title(f"{label}  (n={mask.sum()})")
        ax.set_xlabel("Mean fused σ per sample")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        print(f"    {label:10s}  n={mask.sum():4d}  mean={vals.mean():.4f}  std={vals.std():.4f}")

    axes[0].set_ylabel("Density")
    fig.suptitle(f"Fused σ distribution by K bucket  (overall mean={overall_mean:.4f})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "plot6_sigma_by_K.png", dpi=150)
    plt.close(fig)
    print(f"  → plot6_sigma_by_K.png")
    print("  Interpretation:")
    print("    sigma peaked below 0.45 → sigma_reg needs more weight")
    print("    sigma peaked near  0.45 → sigma_reg working as intended  ✓")


# ── Plot 7 — KL per latent dim ──────────────────────────────────────────────
def plot_kl_per_dim(mu, logvar, out_dir):
    kl = kl_diag_standard_normal(mu, logvar)          # (N, D)
    kl_dim = kl.mean(axis=0)                          # (D,)
    order = np.argsort(kl_dim)[::-1]

    total_kl = kl.sum(axis=1).mean()
    print(f"\n  KL diagnostics:")
    print(f"    Mean total KL per sample: {total_kl:.3f} nats")
    print(f"    Mean KL per dim (top-5):  {', '.join([f'{v:.3f}' for v in kl_dim[order[:5]]])} nats")

    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.bar(np.arange(len(kl_dim)), kl_dim[order], color="#4C72B0", alpha=0.85)
    ax.set_title("KL usage per latent dimension (sorted)")
    ax.set_xlabel("Latent dim (sorted by KL)")
    ax.set_ylabel("Mean KL (nats)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "plot7_kl_per_dim.png", dpi=150)
    plt.close(fig)
    print("  → plot7_kl_per_dim.png")


# ── Plot 8 — KL by K bucket ────────────────────────────────────────────────
def plot_kl_by_K(mu, logvar, K_arr, out_dir):
    kl = kl_diag_standard_normal(mu, logvar)          # (N, D)
    kl_total = kl.sum(axis=1)                         # (N,)
    buckets = _bucket_specs()

    fig, axes = plt.subplots(1, len(buckets), figsize=(14, 4), sharey=True)
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    print(f"\n  KL by K bucket:")
    for ax, (lo, hi, label), col in zip(axes, buckets, colors):
        mask = (K_arr >= lo) & (K_arr < hi)
        vals = kl_total[mask]
        if len(vals) == 0:
            ax.set_title(f"{label}\n(no samples)")
            continue
        ax.hist(vals, bins=35, color=col, alpha=0.85, edgecolor="white", density=True)
        ax.axvline(vals.mean(), color="black", ls="-", lw=1.5, label=f"mean={vals.mean():.2f}")
        ax.set_title(f"{label}  (n={mask.sum()})")
        ax.set_xlabel("Total KL per sample (nats)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        print(f"    {label:10s}  n={mask.sum():4d}  mean={vals.mean():.3f}  std={vals.std():.3f}")

    axes[0].set_ylabel("Density")
    overall_mean = float(kl_total.mean())
    fig.suptitle(f"Total KL distribution by K bucket  (overall mean={overall_mean:.3f} nats)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "plot8_kl_by_K.png", dpi=150)
    plt.close(fig)
    print("  → plot8_kl_by_K.png")


# ── Plot 9 — Linear probe: predict K from mu ───────────────────────────────
def plot_probe_K_from_mu(mu, K_arr, out_dir):
    X_train, X_test, y_train, y_test = train_test_split(
        mu, K_arr.astype(np.float32), test_size=0.2, random_state=42
    )
    reg = LinearRegression(n_jobs=None)
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    r2 = float(r2_score(y_test, pred))
    mae = float(np.mean(np.abs(pred - y_test)))

    print(f"\n  Linear probe K←mu:")
    print(f"    Test R²: {r2:.4f}   MAE: {mae:.3f}")

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(y_test, pred, s=10, alpha=0.35, linewidths=0)
    lo = float(min(y_test.min(), pred.min()))
    hi = float(max(y_test.max(), pred.max()))
    ax.plot([lo, hi], [lo, hi], color="black", lw=1)
    ax.set_title(f"Linear probe: predict K from fused μ\nR²={r2:.3f}  MAE={mae:.2f}")
    ax.set_xlabel("True K")
    ax.set_ylabel("Predicted K")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "plot9_probe_K_from_mu.png", dpi=150)
    plt.close(fig)
    print("  → plot9_probe_K_from_mu.png")


# ── Plot 10 — Expert-vs-fused disagreement vs K ────────────────────────────
def plot_expert_disagreement(experts, fused_mu, fused_logvar, K_arr, out_dir):
    if experts is None:
        print("\n  Expert stats not collected; skipping Plot 10.")
        return

    D = fused_mu.shape[1]
    K_int = K_arr.astype(int)
    K_max = int(K_int.max())
    Ks = np.arange(K_max + 1)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    fig.suptitle("Modality experts vs fused posterior (binned by K)", fontsize=13, fontweight="bold")

    for row, m in enumerate(("heatmap", "occupancy", "impedance")):
        mu_m = experts[m]["mu"]
        lv_m = experts[m]["logvar"]
        sigma_m_mean = np.exp(0.5 * lv_m).mean(axis=1)  # (N,)
        mu_rms_diff = np.sqrt(((mu_m - fused_mu) ** 2).mean(axis=1))  # (N,)

        # Bin stats by exact K value
        counts = np.bincount(K_int, minlength=K_max + 1).astype(np.float32)
        sum_diff = np.bincount(K_int, weights=mu_rms_diff, minlength=K_max + 1)
        sum_diff2 = np.bincount(K_int, weights=mu_rms_diff ** 2, minlength=K_max + 1)
        mean_diff = np.divide(sum_diff, counts, out=np.full_like(sum_diff, np.nan), where=counts > 0)
        var_diff = np.divide(sum_diff2, counts, out=np.full_like(sum_diff2, np.nan), where=counts > 0) - mean_diff ** 2
        std_diff = np.sqrt(np.maximum(var_diff, 0.0))

        sum_sig = np.bincount(K_int, weights=sigma_m_mean, minlength=K_max + 1)
        sum_sig2 = np.bincount(K_int, weights=sigma_m_mean ** 2, minlength=K_max + 1)
        mean_sig = np.divide(sum_sig, counts, out=np.full_like(sum_sig, np.nan), where=counts > 0)
        var_sig = np.divide(sum_sig2, counts, out=np.full_like(sum_sig2, np.nan), where=counts > 0) - mean_sig ** 2
        std_sig = np.sqrt(np.maximum(var_sig, 0.0))

        axd = axes[row, 0]
        axs = axes[row, 1]

        axd.plot(Ks, mean_diff, color="#4C72B0", lw=2)
        axd.fill_between(Ks, mean_diff - std_diff, mean_diff + std_diff, color="#4C72B0", alpha=0.2)
        axd.set_ylabel(f"{m}\nμ RMS diff")
        axd.grid(True, alpha=0.25)

        axs.plot(Ks, mean_sig, color="#55A868", lw=2)
        axs.fill_between(Ks, mean_sig - std_sig, mean_sig + std_sig, color="#55A868", alpha=0.2)
        axs.axhline(0.45, ls="--", color="orange", lw=1.2)
        axs.set_ylabel(f"{m}\nmean σ")
        axs.grid(True, alpha=0.25)

    axes[2, 0].set_xlabel("K")
    axes[2, 1].set_xlabel("K")
    axes[0, 0].set_title("Expert vs fused μ disagreement (mean±std)")
    axes[0, 1].set_title("Expert uncertainty σ (mean±std)")
    fig.tight_layout(rect=[0, 0, 1, 0.96]) # type: ignore
    fig.savefig(out_dir / "plot10_expert_vs_fused_by_K.png", dpi=150)
    plt.close(fig)
    print("\n  Expert-vs-fused diagnostics:")
    print("  → plot10_expert_vs_fused_by_K.png")


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device : {DEVICE}")
    print(f"Output : {out_dir}\n")

    model = load_model(CHECKPOINT_PATH)

    print("\nEncoding dataset…")
    mu, logvar, sigma, K_arr, experts = encode_dataset(model, DATA_DIR, max_samples=MAX_SAMPLES, collect_experts=True)
    print(f"  mu shape: {mu.shape}  K range: [{K_arr.min()}, {K_arr.max()}]")

    plot_tsne(mu, K_arr, out_dir)
    pca, _ = plot_pca_scatter(mu, K_arr, out_dir)
    pca_full = plot_pca_variance(mu, out_dir)
    plot_per_dim(mu, sigma, out_dir)
    plot_decoder_grid(model, pca_full, out_dir, K_val=26, grid_n=8)
    plot_sigma_by_K(sigma, K_arr, out_dir)

    plot_kl_per_dim(mu, logvar, out_dir)
    plot_kl_by_K(mu, logvar, K_arr, out_dir)
    plot_probe_K_from_mu(mu, K_arr, out_dir)
    plot_expert_disagreement(experts, mu, logvar, K_arr, out_dir)

    print("\n" + "=" * 55)
    print("Done.  All plots saved to:", out_dir)
    print("=" * 55)


if __name__ == "__main__":
    main()
