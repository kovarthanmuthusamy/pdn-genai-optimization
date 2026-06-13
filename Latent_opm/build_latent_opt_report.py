"""Build a self-contained HTML report for latent optimization results.

Picks the BEST seed per K (lowest ``best_total`` loss) and embeds
Impedance / Heatmap / Occupancy figures as base64 — open in any browser.

Run:
    python Latent_opm/build_latent_opt_report.py
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import numpy as np
import torch
import torch.nn.functional as F

# =============================================================================
# CONFIG
# =============================================================================

# None → auto-detect the most recently modified run under Latent_opm/runs/
RUN_DIR: str | None = None

# Python module path for the VAE class used in the checkpoint.
# Switch this when reporting a run from a different experiment.
MODEL_MODULE: str = "experiments.exp037_lat_change.codes.vae_multi_input_simple"

# PI_freq value passed to model.decode(). Only used for multi-freq models.
# Ignored if model.decode() does not accept a third argument.
PI_FREQ: float = 0.828

# Asset paths (relative to project root)
FREQUENCY_PATH       = "configs/Frequency_data_hz.npy"
TARGET_IMP_PATH      = "configs/target_impedance.npy"
MASK_PATH            = "configs/binary_mask.npy"
NORM_STATS_PATH      = "datasets/data_norm/normalization_stats.json"

# Output HTML (None → run_dir/latent_opt_report.html)
OUTPUT_HTML: str | None = None

# Plot style
HEATMAP_CMAP   = "jet"
HEATMAP_LEVELS = 22
IMP_YLIM       = (1e-3, 1e2)
DPI            = 200

# Output PNG names (saved inside run_dir/K{xx}/)
PNG_IMP = "best_seed_impedance.png"
PNG_HM  = "best_seed_heatmap.png"
PNG_OCC = "best_seed_occupancy.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float32


# =============================================================================
# Helpers
# =============================================================================

def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@dataclass(frozen=True)
class NormStats:
    imp_log_mean: float
    imp_log_std:  float


@dataclass(frozen=True)
class HmStats:
    hm_log_mean: float
    hm_log_std:  float


def _load_norm_stats(path: Path) -> tuple[NormStats, HmStats]:
    ns = _load_json(path)
    imp = ns["Impedance"]
    hm  = ns["Heatmap"]
    return (
        NormStats(imp_log_mean=float(imp["log_mean"]), imp_log_std=float(imp["log_std"])),
        HmStats(hm_log_mean=float(hm["log_mean"]),    hm_log_std=float(hm["log_std"])),
    )


def _latest_run_dir(runs_root: Path) -> Path:
    candidates = [p for p in runs_root.glob("latent_opt_*") if p.is_dir()]
    if not candidates:
        candidates = [p.parent for p in runs_root.rglob("run_config.json")]
    if not candidates:
        raise SystemExit(f"No runs found under {runs_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _iter_solution_dirs(run_dir: Path) -> list[Path]:
    """All leaf dirs that contain a feasible_best_latent.npy, sorted by (K, seed)."""
    sols: set[Path] = set()
    for p in run_dir.rglob("feasible_best_latent.npy"):
        sols.add(p.parent)

    def _key(d: Path) -> tuple[int, int]:
        k = 999; s = 999999
        m = re.search(r"/K(\d+)(/|$)", d.as_posix())
        if m: k = int(m.group(1))
        ms = re.search(r"seed(\d+)", d.name)
        if ms: s = int(ms.group(1))
        return (k, s)

    return sorted(sols, key=_key)


def _parse_k(sol_dir: Path) -> int:
    m = re.search(r"/K(\d+)(/|$)", sol_dir.as_posix())
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot parse K from: {sol_dir}")


def _best_seed(solutions: list[dict]) -> dict:
    """Return the solution with the lowest peak impedance (max of imp_ohm).

    This picks the physically best result — the seed whose worst-case impedance
    across all frequencies is smallest.  Falls back to first if imp_ohm missing.
    """
    def _score(sol: dict) -> float:
        imp = sol.get("imp_ohm")
        if imp is not None and len(imp) > 0:
            return float(np.max(imp))
        return float("inf")
    return min(solutions, key=_score)


# =============================================================================
# Model decode
# =============================================================================

def _decode(model, z_t: torch.Tensor, K_tensor: torch.Tensor):
    """Call model.decode() with or without PI_freq depending on signature."""
    import inspect
    sig = inspect.signature(model.decode)
    if len(sig.parameters) >= 3:
        pi = torch.full((z_t.shape[0],), PI_FREQ, device=z_t.device, dtype=torch.float32)
        return model.decode(z_t, K_tensor, pi)
    return model.decode(z_t, K_tensor)


# =============================================================================
# Denormalize helpers
# =============================================================================

def _denorm_imp(imp_norm: torch.Tensor, stats: NormStats) -> torch.Tensor:
    """Return blended log-impedance curve, then convert to Ohm."""
    if imp_norm.dim() == 3 and imp_norm.shape[1] >= 2:
        z_raw = imp_norm[:, 0]
        d1    = imp_norm[:, 1]
        z_int = torch.cat([z_raw[:, :1], z_raw[:, :1] + torch.cumsum(d1, dim=-1)[:, :-1]], dim=-1)
        z_log = 0.5 * (z_raw + z_int)
    else:
        z_log = imp_norm.squeeze(1) if imp_norm.dim() == 3 else imp_norm
    z_log_dn = z_log * stats.imp_log_std + stats.imp_log_mean
    return torch.exp(z_log_dn).clamp(min=1e-12)


def _denorm_hm(hm_z: torch.Tensor, stats: HmStats) -> torch.Tensor:
    log1p = hm_z * stats.hm_log_std + stats.hm_log_mean
    return (torch.exp(log1p) - 1.0).clamp(min=0.0)


# =============================================================================
# Decode one solution
# =============================================================================

def _decode_solution(sol_dir: Path, model, stats: NormStats, hm_stats: HmStats) -> dict:
    k_val = _parse_k(sol_dir)
    z_np  = np.load(sol_dir / "feasible_best_latent.npy")
    z_t   = torch.tensor(z_np, device=DEVICE, dtype=DTYPE)
    if z_t.dim() == 1:
        z_t = z_t.unsqueeze(0)
    K_t = torch.full((z_t.shape[0],), k_val, device=z_t.device, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        hm_recon, occ_logits, imp_norm = _decode(model, z_t, K_t)
        occ_prob = torch.sigmoid(occ_logits).cpu().numpy().reshape(-1)
        imp_ohm  = _denorm_imp(imp_norm, stats).cpu().numpy().reshape(-1)
        hm_phys  = _denorm_hm(hm_recon, hm_stats).cpu().numpy().squeeze()
        if hm_phys.ndim == 3:
            hm_phys = hm_phys[0]

    metrics: dict | None = None
    mp = sol_dir / "feasible_best_metrics.json"
    if mp.exists():
        try: metrics = _load_json(mp)
        except Exception: pass

    ms = re.search(r"seed(\d+)", sol_dir.name)
    seed_label = f"seed{ms.group(1)}" if ms else sol_dir.name

    return dict(sol_dir=sol_dir, k_val=k_val, seed_label=seed_label,
                hm=hm_phys, occ_prob=occ_prob, imp_ohm=imp_ohm, metrics=metrics)


# =============================================================================
# Single-seed plots
# =============================================================================

def _plot_impedance(sol: dict, frequency_hz: np.ndarray,
                    target_imp: np.ndarray, out_path: Path) -> Path:
    freq = frequency_hz.reshape(-1)
    tgt  = target_imp.reshape(-1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.loglog(freq, tgt,           "r--", lw=2.0, label="Target")
    ax.loglog(freq, sol["imp_ohm"], color="#1565C0", lw=2.0, label=f"Optimized ({sol['seed_label']})")
    ax.set_ylim(*IMP_YLIM)
    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("Impedance (Ω)",  fontsize=11)
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.4)
    ax.legend(fontsize=10)
    m = sol.get("metrics") or {}
    title = f"K={sol['k_val']:02d}  |  {sol['seed_label']}  |  peak={np.max(sol['imp_ohm']):.4g} Ω"
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_heatmap(sol: dict, mask: np.ndarray | None,
                  cmap, out_path: Path) -> Path:
    hm = sol["hm"]
    if mask is not None:
        try:
            m2 = np.asarray(mask, dtype=bool).reshape(hm.shape)
            hm = np.ma.masked_where(~m2, hm)
        except Exception:
            pass
    vmax = max(float(np.nanmax(hm)) if np.size(hm) else 1.0, 1e-12)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(hm, cmap=cmap, interpolation="bicubic",
                   aspect="auto", origin="lower", vmin=0.0, vmax=vmax)
    ax.set_title(f"K={sol['k_val']:02d}  |  {sol['seed_label']}", fontsize=12)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_ticks(np.linspace(0.0, vmax, 6).tolist())
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_occupancy(sol: dict, out_path: Path) -> Path:
    fig_w = 16.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_w / 4.5))
    _draw_occ_checkboxes(ax, sol["occ_prob"], k_expected=sol["k_val"])
    ax.text(1.0, 1.02, sol["seed_label"], transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color="#555")
    fig.suptitle(f"K={sol['k_val']:02d}  |  Occupancy Probability Grid", fontsize=12, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _draw_occ_checkboxes(ax, occ_prob: np.ndarray, *, k_expected: int) -> None:
    v      = np.asarray(occ_prob, dtype=float).reshape(-1)
    k      = int(k_expected)
    vf     = np.where(np.isfinite(v), v, -np.inf)
    topk_i = np.argsort(-vf, kind="stable")[:k]
    active = np.zeros(52, dtype=bool)
    active[topk_i] = True

    n_rows, n_cols = 4, 13
    cell_w, cell_h = 1.8, 1.3
    pad = 0.07

    ax.set_xlim(0, n_cols * cell_w)
    ax.set_ylim(0, n_rows * cell_h)
    ax.set_aspect("equal")
    ax.axis("off")

    cmap_occ = mcolors.LinearSegmentedColormap.from_list(
        "occ", ["#F5F5F5", "#C8E6C9", "#43A047", "#1B5E20"]
    )

    for idx in range(52):
        r = idx // n_cols
        c = idx % n_cols
        x = c * cell_w
        y = (n_rows - 1 - r) * cell_h
        prob  = float(v[idx]) if np.isfinite(v[idx]) else 0.0
        is_on = bool(active[idx])

        patch = FancyBboxPatch(
            (x + pad, y + pad), cell_w - 2*pad, cell_h - 2*pad,
            boxstyle="round,pad=0.0,rounding_size=0.10",
            facecolor=cmap_occ(prob),
            edgecolor="#1B5E20" if is_on else "#BDBDBD",
            linewidth=2.2 if is_on else 0.6,
            zorder=2,
        )
        ax.add_patch(patch)

        lc = "white" if is_on and prob > 0.55 else "#1A237E" if is_on else "#616161"
        pc = "white" if is_on and prob > 0.65 else "#2E7D32" if is_on else "#9E9E9E"
        ax.text(x + cell_w/2, y + cell_h*0.68, f"C{idx+1}",
                ha="center", va="center", fontsize=5.8,
                fontweight="bold" if is_on else "normal", color=lc, zorder=3)
        ax.text(x + cell_w/2, y + cell_h*0.26, f"{prob:.2f}",
                ha="center", va="center", fontsize=5.4, color=pc, zorder=3)

    ax.set_title(f"Occupancy  —  top-{k} selected", fontsize=9, loc="left", pad=3)


# =============================================================================
# HTML generation (same style as scrap/build_comparison_report.py)
# =============================================================================

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; overflow: hidden; }
body { font-family: Arial, sans-serif; background: #f5f5f5; color: #222; }

#header {
    position: fixed; top: 0; left: 0; right: 0; z-index: 100;
    background: #fff; box-shadow: 0 2px 8px rgba(0,0,0,0.12);
}
h1 { padding: 20px 28px 10px; font-size: 3.5rem; text-align: center; }

.tab-bar {
    display: flex; justify-content: center; gap: 24px; padding: 0 36px;
    border-bottom: 3px solid #ccc; background: #fff;
}
.tab-btn {
    display: flex; flex-direction: column; align-items: center; gap: 6px;
    padding: 18px 48px 14px; border: none;
    border-bottom: 5px solid transparent; background: none;
    cursor: pointer; font-size: 2.2rem; font-weight: 700;
    color: #555; letter-spacing: 0.03em;
    transition: color .15s, border-color .15s;
}
.tab-btn svg { width: 4rem; height: 4rem; stroke: currentColor; fill: none;
    stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; }
.tab-btn:hover { color: #000; }
.tab-btn.active { color: #1a73e8; border-bottom-color: #1a73e8; }

#content { position: fixed; left: 0; right: 0; bottom: 0; overflow: hidden; }

.tab-panel { display: none; position: absolute; top: 0; left: 0; right: 0;
    bottom: 0; overflow-y: auto; padding: 28px; }
.tab-panel.active { display: block; }

.k-section { margin-bottom: 36px; text-align: center; }
.k-section h3 {
    font-size: 2.2rem; font-weight: 700; margin-bottom: 10px;
    padding: 8px 14px; background: #e8f0fe;
    border-left: 5px solid #1a73e8; border-radius: 3px; text-align: center;
}
.k-section img {
    max-width: 100%; border: 1px solid #ddd; border-radius: 4px;
    background: #fff; display: block; margin: 0 auto;
}
.missing { color: #c00; font-style: italic; }

.k-links { display: flex; justify-content: center; gap: 12px; margin: 8px 0 16px; flex-wrap: wrap; }
.k-link {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 18px; font-size: 1.1rem; font-weight: 600;
    color: #1a73e8; background: #e8f0fe;
    border: 1.5px solid #1a73e8; border-radius: 20px;
    cursor: pointer; text-decoration: none; transition: background .15s, color .15s;
}
.k-link:hover { background: #1a73e8; color: #fff; }
"""

_JS = """
function showTab(idx) {
    document.querySelectorAll('.tab-btn').forEach((b,i) => b.classList.toggle('active', i===idx));
    document.querySelectorAll('.tab-panel').forEach((p,i) => p.classList.toggle('active', i===idx));
}
function goTo(tabIdx, anchorId) {
    showTab(tabIdx);
    requestAnimationFrame(() => {
        const el = document.getElementById(anchorId);
        if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
}
function fitContent() {
    const h = document.getElementById('header').offsetHeight;
    document.getElementById('content').style.top = h + 'px';
}
window.addEventListener('load', fitContent);
window.addEventListener('resize', fitContent);
new ResizeObserver(fitContent).observe(document.getElementById('header'));
"""

_ICON_IMP = ('<svg viewBox="0 0 24 24">'
             '<polyline points="2,17 6,11 10,14 14,7 18,10 22,4"/></svg>')
_ICON_HM  = ('<svg viewBox="0 0 24 24">'
             '<path d="M12 2C8 2 5 6 5 10c0 5.25 7 12 7 12s7-6.75 7-12c0-4-3-8-7-8z"/>'
             '<circle cx="12" cy="10" r="2.5"/></svg>')
_ICON_OCC = ('<svg viewBox="0 0 24 24">'
             '<rect x="3" y="3" width="7" height="7" rx="1"/>'
             '<rect x="14" y="3" width="7" height="7" rx="1"/>'
             '<rect x="3" y="14" width="7" height="7" rx="1"/>'
             '<rect x="14" y="14" width="7" height="7" rx="1"/></svg>')


def _img_tag(path: Path, alt: str) -> str:
    if not path.exists():
        return f'<p class="missing">Image not found: {path.name}</p>'
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{alt}">'


def _build_panel(panel_id: str, tab_idx: int,
                 k_img_paths: dict[int, Path],
                 all_tabs: list, active: bool) -> str:
    cls = " active" if active else ""
    parts = [f'<div class="tab-panel{cls}" id="{panel_id}">']
    for k, img_path in sorted(k_img_paths.items()):
        parts.append(f'  <div class="k-section" id="{panel_id}-k{k}">')
        parts.append(f'    <h3>K = {k}</h3>')
        links = []
        for i, (tid, label, icon) in enumerate(all_tabs):
            if i == tab_idx:
                continue
            links.append(
                f'<button class="k-link" onclick="goTo({i},\'{tid}-k{k}\')">'
                f'{icon} {label}</button>'
            )
        parts.append(f'    <div class="k-links">{"".join(links)}</div>')
        parts.append(f'    {_img_tag(img_path, f"K={k}")}')
        parts.append('  </div>')
    parts.append('</div>')
    return "\n".join(parts)


def _build_html(title: str,
                imp_paths: dict[int, Path],
                hm_paths:  dict[int, Path],
                occ_paths: dict[int, Path]) -> str:
    tabs = [
        ("tab-imp", "Impedance", _ICON_IMP),
        ("tab-hm",  "Heatmap",   _ICON_HM),
        ("tab-occ", "Occupancy", _ICON_OCC),
    ]
    path_sets = [imp_paths, hm_paths, occ_paths]

    buttons = "\n".join(
        f'<button class="tab-btn{" active" if i==0 else ""}" onclick="showTab({i})">'
        f'{icon}<span>{label}</span></button>'
        for i, (_, label, icon) in enumerate(tabs)
    )
    panels = "\n".join(
        _build_panel(tid, i, paths, tabs, active=(i == 0))
        for i, ((tid, _, _icon), paths) in enumerate(zip(tabs, path_sets))
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{_CSS}</style>
</head>
<body>
<div id="header">
<h1>{title}</h1>
<div class="tab-bar">
{buttons}
</div>
</div>
<div id="content">
{panels}
</div>
<script>{_JS}</script>
</body>
</html>
"""


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    repo_root = _project_root()
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # ── Resolve run dir ───────────────────────────────────────────────────────
    run_dir = Path(RUN_DIR).resolve() if RUN_DIR else _latest_run_dir(repo_root / "Latent_opm" / "runs")
    if not run_dir.exists():
        raise SystemExit(f"RUN_DIR not found: {run_dir}")
    print(f"Run folder: {run_dir}")

    run_cfg_path = run_dir / "run_config.json"
    if not run_cfg_path.exists():
        raise SystemExit(f"Missing run_config.json in: {run_dir}")
    run_cfg = _load_json(run_cfg_path)

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt_rel  = run_cfg.get("checkpoint", "")
    ckpt_path = (repo_root / ckpt_rel).resolve()
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    print(f"Checkpoint: {ckpt_path}")

    import importlib
    mod = importlib.import_module(MODEL_MODULE)
    MultiInputVAE = mod.MultiInputVAE

    device = torch.device(DEVICE)
    ckpt   = torch.load(ckpt_path, map_location=device)
    cfg    = ckpt.get("config", {})
    latent_dim          = int(run_cfg.get("latent_dim",          cfg.get("latent_dim",          32)))
    cond_dim            = int(run_cfg.get("cond_dim",            cfg.get("cond_dim",             8)))
    heatmap_private_dim = int(run_cfg.get("heatmap_private_dim", cfg.get("heatmap_private_dim",  8)))

    model = MultiInputVAE(
        latent_dim=latent_dim, cond_dim=cond_dim,
        heatmap_private_dim=heatmap_private_dim, modality_dropout=0.0,
    ).to(device=device, dtype=DTYPE)

    # Shape-safe partial load
    ckpt_sd   = ckpt.get("model_state_dict", ckpt)
    model_sd  = model.state_dict()
    compat    = {k: v for k, v in ckpt_sd.items() if k in model_sd and v.shape == model_sd[k].shape}
    model_sd.update(compat)
    model.load_state_dict(model_sd)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Model loaded ({len(compat)}/{len(model_sd)} tensors matched)")

    # ── Load assets ───────────────────────────────────────────────────────────
    frequency_hz = np.load(repo_root / FREQUENCY_PATH).reshape(-1).astype(np.float64)
    target_imp   = np.load(repo_root / TARGET_IMP_PATH).reshape(-1).astype(np.float64)
    mask_p = repo_root / MASK_PATH
    mask   = np.load(mask_p).astype(bool) if mask_p.exists() else None

    if isinstance(run_cfg.get("normalization_stats"), dict):
        ns = run_cfg["normalization_stats"]
        stats   = NormStats(imp_log_mean=float(ns["imp_log_mean"]), imp_log_std=float(ns["imp_log_std"]))
        hm_stat_p = repo_root / NORM_STATS_PATH
        _, hm_stats = _load_norm_stats(hm_stat_p) if hm_stat_p.exists() else (None, HmStats(0.0, 1.0))
    else:
        stats, hm_stats = _load_norm_stats(repo_root / NORM_STATS_PATH)

    cmap = mpl.colormaps[HEATMAP_CMAP].resampled(HEATMAP_LEVELS).copy()
    cmap.set_bad("white")

    # ── Decode all solutions, group by K ─────────────────────────────────────
    sol_dirs = _iter_solution_dirs(run_dir)
    if not sol_dirs:
        raise SystemExit(f"No solutions found under: {run_dir}")
    print(f"Solutions found: {len(sol_dirs)}")

    by_k: dict[int, list[dict]] = defaultdict(list)
    for sd in sol_dirs:
        try:
            data = _decode_solution(sd, model, stats, hm_stats)
            by_k[data["k_val"]].append(data)
        except Exception as e:
            print(f"  Warning: skipping {sd.name}: {e}")

    # ── Pick best seed per K, generate PNGs ──────────────────────────────────
    imp_paths: dict[int, Path] = {}
    hm_paths:  dict[int, Path] = {}
    occ_paths: dict[int, Path] = {}

    for k_val in sorted(by_k):
        sols = by_k[k_val]
        best = _best_seed(sols)
        m    = best.get("metrics") or {}

        # Resolve K output dir (same as where latent opt saved results)
        k_dir: Path | None = None
        for part in best["sol_dir"].relative_to(run_dir).parts:
            if re.fullmatch(r"K\d+", part):
                k_dir = run_dir / part
                break
        if k_dir is None:
            k_dir = run_dir / f"K{k_val:02d}"

        peak_ohm = float(np.max(best["imp_ohm"])) if best["imp_ohm"] is not None else float("nan")
        print(f"  K={k_val:02d}  best={best['seed_label']}  peak_imp={peak_ohm:.4g} Ω  ({len(sols)} feasible)")

        imp_paths[k_val] = _plot_impedance(best, frequency_hz, target_imp, k_dir / PNG_IMP)
        hm_paths[k_val]  = _plot_heatmap(best, mask, cmap,                  k_dir / PNG_HM)
        occ_paths[k_val] = _plot_occupancy(best,                             k_dir / PNG_OCC)

    # ── Build HTML ────────────────────────────────────────────────────────────
    run_name = run_dir.name
    out_path = Path(OUTPUT_HTML).resolve() if OUTPUT_HTML else run_dir / "latent_opt_report.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    html = _build_html(f"Latent Opt Report — {run_name}", imp_paths, hm_paths, occ_paths)
    out_path.write_text(html, encoding="utf-8")
    print(f"\nReport: {out_path}")
    print("Open in any browser — no dependencies needed.")


if __name__ == "__main__":
    main()
