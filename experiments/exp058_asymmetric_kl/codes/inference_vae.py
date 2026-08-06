"""
Inference script for Multi-Input VAE — exp052.

Reads data_dir / background from this experiment's config.yaml (datasets/data_multifreq_train_norm_robust).
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.widgets import Button

import sys
from pathlib import Path

_REPO_BOOT = Path(__file__).resolve().parents[3]
if str(_REPO_BOOT) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOT))

from repo_paths import REPO_ROOT as PROJECT_ROOT, resolve_repo_path, setup_path
setup_path()

_EXP_DIR = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _EXP_DIR / "config.yaml"


def load_experiment_config(path: Path) -> dict:
    """Parse config.yaml: JSON object with optional full-line ``#`` comments."""
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(line)
    return json.loads("\n".join(lines))


def _default_data_dir() -> Path:
    from libs.experiment_paths import data_dir_from_config

    return data_dir_from_config(_CONFIG_PATH)


def _norm_stats_path() -> Path:
    from libs.experiment_paths import norm_stats_path

    return norm_stats_path(_default_data_dir())


from experiments.exp058_asymmetric_kl.codes.vae_poe_freq import MultiInputVAEPoeFreq  # noqa: E402
from libs.data_creation.csv_to_occupancy import labels_v1  # noqa: E402
from src_vae.others.heatmap_z_clip import load_heatmap_z_clip_bounds  # noqa: E402
from src_vae.others.norm_stats import HeatmapNormStats, NormStatsBundle, load_norm_stats  # noqa: E402

# ============================================================
# CONFIGURATION (standalone CLI defaults)
# ============================================================
CHECKPOINT_PATH = str(_EXP_DIR / "checkpoints/last_model.pt")
LATENT_STATS_PATH = str(_EXP_DIR / "metrics/latent_stats.json")
MODEL_LATENT_DIM = 48

NUM_SAMPLES = 3
K_VALUE = 5
PI_FREQ_MHZ = 200.0
OUTPUT_DIR = str(_EXP_DIR / "visuals_K5")
SAVE_DATA = True
SAVE_PLOTS = True
USE_CUDA = True
SHARED_TEMP = 1.5


class VAEInference:
    """Inference engine for exp055 multifreq PI heatmap VAE."""

    def __init__(
        self,
        checkpoint_path: str,
        latent_dim: int = MODEL_LATENT_DIM,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.latent_dim = latent_dim

        stats_path = _norm_stats_path()
        self.norm_stats: NormStatsBundle = load_norm_stats(_default_data_dir(), stats_path=stats_path)
        hm = self.norm_stats.heatmap
        imp = self.norm_stats.impedance
        self.hm_stats: HeatmapNormStats = hm
        self.imp_log_mean = imp.log_mean
        self.imp_log_std = imp.log_std
        self.hm_log_mean = hm.log_mean
        self.hm_log_std = hm.log_std
        clip = load_heatmap_z_clip_bounds(_default_data_dir(), stats_path=stats_path)
        self.hm_z_clip = clip
        self.background_value = float(
            self.norm_stats.raw.get(
                "background_value",
                hm.background_value,
            )
        )
        if _CONFIG_PATH.is_file():
            cfg = load_experiment_config(_CONFIG_PATH)
            self.background_value = float(cfg.get("background_value", self.background_value))

        print(f"Norm stats: {stats_path}")
        print(f"  heatmap: {hm.describe()}")
        print(f"Impedance (log): mean={self.imp_log_mean:.4f}, std={self.imp_log_std:.4f}")
        if clip is not None:
            print(f"  clip bounds (widest): [{clip[0]:.4f}, {clip[1]:.4f}]")

        cfg_dir = PROJECT_ROOT / "configs"
        self.frequency = np.load(cfg_dir / "Frequency_data_hz.npy").squeeze()
        self.target_impedance = np.load(cfg_dir / "target_impedance.npy").squeeze()
        self.binary_mask = np.load(cfg_dir / "binary_mask.npy")

        print(f"\nLoading: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg = ckpt.get("config", {})
        ld = cfg.get("latent_dim", latent_dim)
        cond = cfg.get("cond_dim", 8)
        hm_priv = cfg.get("heatmap_private_dim", 8)
        mod_drop = cfg.get("modality_dropout", 0.0)
        print(
            f"Model: latent_dim={ld}  cond_dim={cond}  heatmap_private_dim={hm_priv}  "
            f"modality_dropout={mod_drop}"
        )

        self.model = MultiInputVAEPoeFreq(
            latent_dim=ld,
            cond_dim=cond,
            heatmap_private_dim=hm_priv,
            modality_dropout=mod_drop,
            freq_fourier_features=int(cfg.get("freq_fourier_features", 8)),
            use_heatmap_film=bool(cfg.get("use_heatmap_film", True)),
            use_freq_poe_expert=bool(cfg.get("use_freq_poe_expert", True)),
            use_heatmap_unet_skips=bool(cfg.get("use_heatmap_unet_skips", True)),
            use_occ_spatial_decoder=bool(cfg.get("use_occ_spatial_decoder", True)),
            use_occ_spatial_tower=bool(cfg.get("use_occ_spatial_tower", True)),
            occ_spatial_ch=int(cfg.get("occ_spatial_ch", 8)),
            use_layout_private_head=bool(cfg.get("use_layout_private_head", True)),
            layout_private_hidden=int(cfg.get("layout_private_hidden", 384)),
            use_layout_private_freq_film=bool(cfg.get("use_layout_private_freq_film", True)),
        )

        ckpt_state = ckpt["model_state_dict"]
        model_state = self.model.state_dict()
        compat = {
            k: v for k, v in ckpt_state.items()
            if k in model_state and model_state[k].shape == v.shape
        }
        skipped = [k for k in ckpt_state if k not in compat]
        load_result = self.model.load_state_dict(compat, strict=False)

        required_prefixes = (
            "heatmap_fc.",
            "heatmap_dec_deconv1.",
            "heatmap_dec_deconv2.",
            "occupancy_decoder.",
            "impedance_decoder.",
            "heatmap_mu.",
            "occupancy_mu.",
            "impedance_mu.",
            "freq_poe_mu.",
            "layout_private_mu.",
        )
        missing = [k for k in load_result.missing_keys if k.startswith(required_prefixes)]
        if missing:
            raise RuntimeError(f"Missing generation-critical weights: {missing[:12]}")

        print(f"Loaded {len(compat)}/{len(ckpt_state)} tensors")
        if skipped:
            print(f"Skipped {len(skipped)}: {', '.join(skipped[:8])}{'...' if len(skipped) > 8 else ''}")

        self.model.to(self.device).eval()
        self.latent_stats = ckpt.get("latent_stats", None)
        self.per_K_latent_stats = ckpt.get("per_K_latent_stats", None)

    def load_latent_stats(self, latent_stats_path: Optional[str] = None):
        if latent_stats_path is None:
            auto = Path(self.checkpoint_path).parent.parent / "metrics" / "latent_stats.json"
            if auto.exists():
                latent_stats_path = str(auto)

        if latent_stats_path and Path(latent_stats_path).exists():
            with open(latent_stats_path, encoding="utf-8") as f:
                self.latent_stats = json.load(f)
            print(f"Loaded latent stats: {latent_stats_path}")
            s = self.latent_stats.get("latent", {})
            print(
                f"  latent: mu_mean={s.get('mu_mean', '?'):.4f}, "
                f"sigma_mean={s.get('sigma_mean', '?'):.4f}"
            )
        elif self.latent_stats:
            print("Using latent stats embedded in checkpoint.")
        else:
            print("WARNING: No latent stats — sampling from N(0,1).")

    def _denorm_impedance(self, imp_norm):
        z_raw = imp_norm[:, 0] if imp_norm.dim() == 3 else imp_norm
        return z_raw * self.imp_log_std + self.imp_log_mean

    def fg_threshold(self) -> float:
        """Foreground mask threshold in model train space (log z-score)."""
        return float(self.background_value) + 0.5

    def denorm_heatmap_physical(
        self,
        hm_norm: torch.Tensor,
        *,
        mhz: float | None = None,
        pi_norm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Denorm model heatmap output to physical Ω (robust per-MHz or legacy z-score)."""
        return self.hm_stats.norm_to_physical(
            hm_norm,
            mhz=mhz,
            pi_norm=pi_norm,
            apply_clip=not self.hm_stats.is_unbounded(),
        )

    def generate_save(self, num_samples=1, out_dir="temp_visuals", K=26, PI_freq_mhz=200.0):
        with torch.no_grad():
            hm_z, occ, imp_norm = self.model.inference(
                num_samples,
                self.device,
                K=K,
                PI_freq=PI_freq_mhz,
                pi_freq_unit="mhz",
                latent_stats=self.latent_stats,
                per_K_latent_stats=self.per_K_latent_stats,
                shared_temp=SHARED_TEMP,
            )
            occ_bin = torch.zeros_like(occ)
            if K > 0:
                topk_idx = occ.topk(min(K, occ.shape[-1]), dim=-1).indices
                occ_bin.scatter_(-1, topk_idx, 1.0)
            imp = self._denorm_impedance(imp_norm)
            hm_phys = self.denorm_heatmap_physical(hm_z, mhz=PI_freq_mhz)

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        plots = out / "plots"
        plots.mkdir(exist_ok=True)

        hm_z_np = hm_z.cpu().numpy()
        hm_phys_np = hm_phys.cpu().numpy()
        occ_prob_np = occ.cpu().numpy()
        occ_np = occ_bin.cpu().numpy()
        imp_np = imp.cpu().numpy()

        print(f"\nShapes — heatmap: {hm_phys_np.shape}, occ: {occ_np.shape}, imp: {imp_np.shape}")
        print(
            f"Ranges — heatmap: [{hm_phys_np.min():.4f},{hm_phys_np.max():.4f}], "
            f"imp: [{imp_np.min():.4f},{imp_np.max():.4f}]"
        )

        def _process_one(i):
            d = out / f"data_sample_{i}"
            d.mkdir(exist_ok=True)
            for fname, arr in {
                "heatmap_zscore.npy": hm_z_np[i],
                "heatmap_physical.npy": hm_phys_np[i],
                "occupancy_map.npy": occ_np[i],
                "impedance_profile.npy": imp_np[i],
            }.items():
                np.save(d / fname, arr)
            self.visualize_sample(
                heatmap_physical=hm_phys_np[i],
                impedance_log=imp_np[i],
                occupancy=occ_np[i],
                occupancy_prob=occ_prob_np[i],
                sample_idx=i,
                output_dir=plots,
                K=K,
            )

        with ThreadPoolExecutor(max_workers=min(num_samples, os.cpu_count() or 4)) as ex:
            list(ex.map(_process_one, range(num_samples)))

        plot_paths = sorted(
            plots.glob("visualization_sample_*.png"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        print(f"\n✓ {num_samples} plots saved to: {plots}")
        launch_interactive_viewer(plot_paths)

    def visualize_sample(
        self,
        heatmap_physical,
        impedance_log,
        occupancy,
        sample_idx,
        output_dir,
        K=None,
        occupancy_prob=None,
    ):
        fig = Figure(figsize=(22, 7))
        FigureCanvasAgg(fig)

        ax1 = fig.add_subplot(1, 3, 1)
        hm = np.ma.masked_where(~self.binary_mask, heatmap_physical[0])
        cmap = mpl.colormaps["jet"].resampled(22).copy()
        cmap.set_bad("white")
        im = ax1.imshow(
            hm,
            cmap=cmap,
            interpolation="bicubic",
            aspect="auto",
            origin="lower",
            vmin=0,
            vmax=hm.max(),
        )
        ax1.set_title(f"Generated Heatmap (Sample {sample_idx})", fontsize=14)
        cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        cb.set_ticks(np.linspace(*im.get_clim(), 6).tolist())

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.loglog(self.frequency, self.target_impedance, "--", lw=2.5, label="Target", color="red")
        ax2.loglog(self.frequency, np.exp(impedance_log), "-", lw=2.5, color="royalblue", label="Generated")
        ax2.set_ylim(1e-3, 1e2)
        ax2.grid(True, which="both", alpha=0.3)
        ax2.legend(fontsize=9)

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.axis("off")
        occ_vec = occupancy.flatten()
        prob_vec = occupancy_prob.flatten() if occupancy_prob is not None else occ_vec
        n, half = len(occ_vec), (len(occ_vec) + 1) // 2
        ACT, INACT, HDR = "#C8E6C9", "#F5F5F5", "#90CAF9"
        rows, cell_colors = [], []
        for r in range(half):
            row, col = [], []
            for side in (0, 1):
                idx = r + side * half
                if idx < n:
                    active = float(occ_vec[idx]) > 0.5
                    row += [f"{float(prob_vec[idx]):.3f}", labels_v1[idx] if active else ""]
                    col += [ACT if active else INACT] * 2
                else:
                    row += ["", ""]
                    col += [INACT, INACT]
            rows.append(row)
            cell_colors.append(col)

        tbl = ax3.table(
            cellText=rows,
            colLabels=["Raw", "Label", "Raw", "Label"],
            cellColours=cell_colors,
            colColours=[HDR] * 4,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.scale(1.0, 0.82)
        k_str = f"  K={K}" if K is not None else ""
        ax3.set_title(
            f"Occupancy  ({int((occ_vec > 0.5).sum())}/52 active){k_str}  —  Sample {sample_idx}",
            fontsize=12,
        )

        fig.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.12, wspace=0.38)
        out_path = Path(output_dir) / f"visualization_sample_{sample_idx}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Visualization saved: {out_path}")


def launch_interactive_viewer(plot_paths):
    if not plot_paths:
        return
    for backend in ("TkAgg", "Qt5Agg", "Qt6Agg", "WebAgg"):
        try:
            plt.switch_backend(backend)
            break
        except Exception:
            continue

    plot_paths = list(plot_paths)
    images = [plt.imread(str(p)) for p in plot_paths]
    n, state = len(plot_paths), {"idx": 0}

    fig, ax = plt.subplots(figsize=(18, 6))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")
    plt.subplots_adjust(bottom=0.13)

    h, w = images[0].shape[:2]
    img_obj = ax.imshow(images[0], extent=(0, w, h, 0))
    ax.axis("off")
    title_obj = ax.set_title("", color="white", fontsize=13, pad=8)

    def _update(idx):
        state["idx"] = idx % n
        img = images[state["idx"]]
        img_obj.set_data(img)
        img_obj.set_extent((0, img.shape[1], img.shape[0], 0))
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
        title_obj.set_text(
            f"Sample {state['idx'] + 1}/{n}  [{plot_paths[state['idx']].name}]  "
            "← → navigate | Esc exit"
        )
        fig.canvas.draw_idle()

    _update(0)

    ax_prev = plt.axes([0.36, 0.02, 0.12, 0.07])  # type: ignore
    ax_next = plt.axes([0.52, 0.02, 0.12, 0.07])  # type: ignore
    btn_prev = Button(ax_prev, "◄  Prev", color="#333333", hovercolor="#555555")
    btn_next = Button(ax_next, "Next  ►", color="#333333", hovercolor="#555555")
    btn_prev.label.set_color("white")
    btn_next.label.set_color("white")
    btn_prev.on_clicked(lambda _e: _update(state["idx"] - 1))
    btn_next.on_clicked(lambda _e: _update(state["idx"] + 1))

    def _on_key(event):
        if event.key == "right":
            _update(state["idx"] + 1)
        elif event.key == "left":
            _update(state["idx"] - 1)
        elif event.key == "escape":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", _on_key)
    plt.show()


def main():
    device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
    print(f"Checkpoint: {CHECKPOINT_PATH} | Samples: {NUM_SAMPLES} | K={K_VALUE} | Device: {device}")
    engine = VAEInference(checkpoint_path=CHECKPOINT_PATH, latent_dim=MODEL_LATENT_DIM, device=device)
    engine.load_latent_stats(LATENT_STATS_PATH)
    engine.generate_save(
        out_dir=OUTPUT_DIR,
        num_samples=NUM_SAMPLES,
        K=K_VALUE,
        PI_freq_mhz=PI_FREQ_MHZ,
    )
    print("\n✓ Inference complete!")


if __name__ == "__main__":
    main()
