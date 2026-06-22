"""Occupancy checkbox plot for one K.

Purpose:
    Visualize generated 52-slot occupancy vectors (C1..C52) for one K using top-K or threshold policy.

Run:
    python scrap/comparison/compare_generated_vs_real_occupancy.py

Agent notes:
    - What: Renders which decap slots are active per generated sample (bar/checkbox view).
    - Usage: Set ``K_VALUE``, ``BASE_GENERATED_DIR``, ``ACTIVE_POLICY`` → run.
    - Config keys:
        - ``K_VALUE``, ``BASE_GENERATED_DIR`` — which K folder to read
        - ``NUM_SAMPLES`` — rows to plot
        - ``ACTIVE_POLICY`` — ``"topk"`` or ``"threshold"``; ``THRESHOLD`` when threshold mode
        - ``OCCUPANCY_OUT_NAME`` — output PNG name
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch
import numpy as np


# =============================================================================
# CONFIGURATION — edit these before running: python scrap/comparison/compare_generated_vs_real_occupancy.py
# =============================================================================
K_VALUE = 1
BASE_GENERATED_DIR = Path("scrap/generated_samples")

# If None, infer from data_sample_* folders.
NUM_SAMPLES: int | None = None

# Output (saved under the K folder)
OCCUPANCY_OUT_NAME = "generated_occupancy.png"

# Activation policy:
# - "topk": mark exactly K entries active (highest values), matching inference_vae.py
# - "threshold": mark entries active if value > THRESHOLD
ACTIVE_POLICY: str = "topk"
THRESHOLD: float = 0.5

# =============================================================================


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _infer_num_samples(k_dir: Path) -> int:
    sample_dirs = [p for p in k_dir.glob("data_sample_*") if p.is_dir()]
    if not sample_dirs:
        raise SystemExit(f"No data_sample_* folders found in: {k_dir}")

    def idx(p: Path) -> int | None:
        name = p.name
        if not name.startswith("data_sample_"):
            return None
        try:
            return int(name.split("_")[-1])
        except Exception:
            return None

    indices = sorted(i for i in (idx(p) for p in sample_dirs) if i is not None)
    if not indices:
        raise SystemExit(f"Found data_sample_* entries but none matched expected naming in: {k_dir}")

    expected = list(range(0, max(indices) + 1))
    if indices != expected:
        raise SystemExit(
            "data_sample_* folders are not consecutive starting at 0.\n"
            f"  Found:    {indices[:20]}{' ...' if len(indices) > 20 else ''}\n"
            f"  Expected: {expected[:20]}{' ...' if len(expected) > 20 else ''}"
        )

    return len(expected)


def _load_occupancy_matrix(k_dir: Path, *, num_samples: int) -> np.ndarray:
    """Load generated occupancy vectors for all samples under a K folder."""
    occ_path = k_dir / "occupancy.npy"
    if occ_path.exists():
        occ = np.asarray(np.load(occ_path))
        if occ.ndim == 1:
            if occ.shape[0] != 52:
                raise ValueError(f"Expected occupancy vector length 52 in {occ_path}, got shape {occ.shape}")
            return occ.reshape(1, 52)
        if occ.ndim == 2 and occ.shape[1] == 52:
            return occ
        raise ValueError(f"Expected occupancy.npy shape (N,52) or (52,), got {occ.shape} from {occ_path}")

    rows: list[np.ndarray] = []
    for i in range(num_samples):
        p = k_dir / f"data_sample_{i}" / "occupancy_map.npy"
        if not p.exists():
            raise SystemExit(f"No occupancy.npy and missing per-sample occupancy_map.npy: {p}")
        v = np.asarray(np.load(p)).reshape(-1)
        if v.shape[0] != 52:
            raise ValueError(f"Expected occupancy_map length 52 in {p}, got shape {v.shape}")
        rows.append(v)
    return np.stack(rows, axis=0)


def _plot_checkboxes(
    *,
    occupancy_matrix: np.ndarray,
    out_path: Path,
    title_prefix: str,
    expected_k: int | None = None,
) -> None:
    """Render compact checkbox grids labeled C1..C52."""
    occ = np.asarray(occupancy_matrix)
    if occ.ndim != 2 or occ.shape[1] != 52:
        raise ValueError(f"Expected occupancy_matrix shape (N,52), got {occ.shape}")

    if ACTIVE_POLICY not in {"topk", "threshold"}:
        raise ValueError(f"ACTIVE_POLICY must be 'topk' or 'threshold', got {ACTIVE_POLICY!r}")

    if expected_k is not None and (expected_k < 0 or expected_k > 52):
        raise ValueError(f"expected_k must be in [0,52], got {expected_k}")

    def _active_mask(v: np.ndarray) -> np.ndarray:
        vf = np.asarray(v, dtype=float).reshape(-1)
        if vf.shape[0] != 52:
            raise ValueError(f"Expected occupancy vector length 52, got {vf.shape}")

        if ACTIVE_POLICY == "threshold" or expected_k is None:
            return vf > float(THRESHOLD)

        k = int(expected_k)
        if k <= 0:
            return np.zeros(52, dtype=bool)
        if k >= 52:
            return np.ones(52, dtype=bool)

        vf = np.where(np.isfinite(vf), vf, -np.inf)
        # Activate exactly K highest-probability slots (ties broken by sort order).
        topk_idx = np.argsort(-vf, kind="stable")[:k]
        mask = np.zeros(52, dtype=bool)
        mask[topk_idx] = True
        return mask

    n = occ.shape[0]
    fig_h = max(2.2, 1.25 * n)
    fig, axes = plt.subplots(n, 1, figsize=(14, fig_h))
    if isinstance(axes, Axes):
        axes_list: list[Axes] = [axes]
    else:
        axes_list = list(axes)

    # Subtle, clean palette
    active_face = "#C8E6C9"   # light green (darker than before)
    inactive_face = "#FAFAFA" # near-white
    inactive_text = "#9E9E9E"
    edge_color = "#BDBDBD"

    n_rows, n_cols = 4, 13
    cell_w, cell_h = 1.0, 1.0

    for i, ax in enumerate(axes_list):
        v = occ[i].reshape(-1)
        active = _active_mask(v)
        k_active = int(active.sum())

        ax.set_xlim(0, n_cols * cell_w)
        ax.set_ylim(0, n_rows * cell_h)
        ax.set_aspect("equal")
        ax.axis("off")

        for idx in range(52):
            r = idx // n_cols
            c = idx % n_cols
            y = (n_rows - 1 - r) * cell_h
            x = c * cell_w
            is_on = bool(active[idx])
            fc = active_face if is_on else inactive_face

            # Rounded checkbox
            patch = FancyBboxPatch(
                (x, y),
                cell_w,
                cell_h,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                facecolor=fc,
                edgecolor=edge_color,
                linewidth=0.9,
            )
            ax.add_patch(patch)

            # Label (always shown)
            ax.text(
                x + cell_w / 2,
                y + 0.22,
                f"C{idx + 1}",
                ha="center",
                va="center",
                fontsize=6.5,
                color=("#212121" if is_on else inactive_text),
            )

        policy = "topK" if ACTIVE_POLICY == "topk" and expected_k is not None else f"> {THRESHOLD:g}"
        k_note = (
            f"  (policy={policy}, active {k_active}/52"
            + (f", expected K={expected_k}" if expected_k is not None else "")
            + ")"
        )
        ax.set_title(f"{title_prefix}sample_{i}{k_note}", fontsize=9.5, pad=3, loc="left")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved occupancy plot: {out_path}")


def main() -> None:
    repo_root = _project_root()
    os.chdir(repo_root)

    k_dir = (repo_root / BASE_GENERATED_DIR / f"K{K_VALUE}").resolve()
    if not k_dir.exists():
        raise SystemExit(f"K folder not found: {k_dir}")

    num_samples = NUM_SAMPLES if NUM_SAMPLES is not None else _infer_num_samples(k_dir)

    print(f"K folder: {k_dir}")
    print(f"Samples:  {num_samples}")

    occ = _load_occupancy_matrix(k_dir, num_samples=num_samples)
    out_path = k_dir / OCCUPANCY_OUT_NAME

    _plot_checkboxes(
        occupancy_matrix=occ,
        out_path=out_path,
        title_prefix=f"K{K_VALUE}/",
        expected_k=K_VALUE,
    )


if __name__ == "__main__":
    main()
