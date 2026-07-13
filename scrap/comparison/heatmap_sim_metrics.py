"""Metrics for generated vs ECADStar-simulated real heatmaps (post-move).

Compares ``heatmap_physical.npy`` (generated) against interpolated ``Z_*MHz.map``
from each ``K{n}/Real/`` folder — not dataset val heatmaps.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SIM_METRICS_CSV = "sim_compare_metrics.csv"
SIM_METRICS_JSON = "sim_compare_metrics.json"
SIM_METRICS_MD = "sim_compare_metrics.md"

# Primary QC columns for MD tables / agent summaries (see run_multifreq_sweep_pipeline.py).
# Prefer pattern + scale over mae_ohm (often 0 at 10 MHz FG) or mape_pct (near-zero divide).
PRIMARY_METRIC_COLUMNS: tuple[str, ...] = (
    "mhz",
    "k",
    "sample",
    "pearson_r",
    "max_diff_ohm",
    "pattern_mae",
    "max_ratio",
    "peak_loc_err_px",
    "real_max_ohm",
    "gen_max_ohm",
)
PRIMARY_BY_MHZ_COLUMNS: tuple[str, ...] = (
    "mhz",
    "n",
    "pearson_r_mean",
    "max_diff_ohm_mean",
    "pattern_mae_mean",
    "max_ratio_mean",
    "peak_loc_err_px_mean",
    "real_max_ohm_mean",
    "gen_max_ohm_mean",
)

METRIC_COLUMNS: tuple[str, ...] = (
    "mhz",
    "k",
    "sample",
    "label",
    "mae_ohm",
    "rmse_ohm",
    "median_abs_diff_ohm",
    "p95_abs_diff_ohm",
    "max_abs_diff_ohm",
    "nrmse_ohm",
    "mape_pct",
    "real_max_ohm",
    "gen_max_ohm",
    "max_diff_ohm",
    "real_p95_ohm",
    "gen_p95_ohm",
    "real_mean_ohm",
    "gen_mean_ohm",
    "max_ratio",
    "mean_ratio",
    "pearson_r",
    "spearman_rho",
    "pattern_mae",
    "pattern_rmse",
    "peak_loc_err_px",
    "peak_real_ohm",
    "peak_gen_ohm",
)


def _peak_location_2d(arr: np.ndarray, mask: np.ndarray) -> tuple[int, int]:
    """Argmax (row, col) within mask; ties → first occurrence."""
    masked = np.where(mask, arr, -np.inf)
    flat_idx = int(np.argmax(masked))
    return divmod(flat_idx, arr.shape[1])


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    try:
        from scipy.stats import spearmanr

        return float(spearmanr(x, y).statistic)
    except Exception:
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1])


def compute_sim_heatmap_metrics(
    real: np.ndarray,
    gen: np.ndarray,
    mask: np.ndarray,
    *,
    mhz: int | float | None = None,
    k: int | None = None,
    sample: int | None = None,
    label: str = "",
    diff_tolerance: float = 0.0,
    pattern_diff_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Full error metrics between simulated-real and generated heatmaps (Ω, masked FG)."""
    real = np.asarray(real, dtype=np.float64)
    gen = np.asarray(gen, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)

    real_vals = real[mask]
    gen_vals = gen[mask]
    if real_vals.size == 0:
        raise ValueError("mask selects zero foreground pixels")

    diff = np.abs(real - gen)
    if diff_tolerance > 0:
        diff = np.where(diff < diff_tolerance, 0.0, diff)
    diff_fg = diff[mask]

    real_norm = np.zeros_like(real, dtype=np.float64)
    gen_norm = np.zeros_like(gen, dtype=np.float64)
    r_rng = max(float(real_vals.max() - real_vals.min()), 1e-12)
    g_rng = max(float(gen_vals.max() - gen_vals.min()), 1e-12)
    real_norm[mask] = (real_vals - real_vals.min()) / r_rng
    gen_norm[mask] = (gen_vals - gen_vals.min()) / g_rng
    pattern_diff = np.abs(real_norm - gen_norm)
    if pattern_diff_tolerance > 0:
        pattern_diff[mask] = np.where(
            pattern_diff[mask] < pattern_diff_tolerance,
            0.0,
            pattern_diff[mask],
        )
    pat_fg = pattern_diff[mask]

    mae = float(np.mean(diff_fg))
    rmse = float(np.sqrt(np.mean(diff_fg**2)))
    real_max = float(real_vals.max())
    gen_max = float(gen_vals.max())
    max_diff_ohm = real_max - gen_max
    real_mean = float(np.mean(real_vals))
    gen_mean = float(np.mean(gen_vals))
    real_p95 = float(np.percentile(real_vals, 95))
    gen_p95 = float(np.percentile(gen_vals, 95))

    pr_row, pr_col = _peak_location_2d(real, mask)
    pg_row, pg_col = _peak_location_2d(gen, mask)
    peak_dist = float(np.hypot(pr_row - pg_row, pr_col - pg_col))

    pearson_r = float(np.corrcoef(real_vals, gen_vals)[0, 1]) if real_vals.size > 1 else float("nan")

    row: dict[str, Any] = {
        "mhz": float(mhz) if mhz is not None else None,
        "k": int(k) if k is not None else None,
        "sample": int(sample) if sample is not None else None,
        "label": label,
        "mae_ohm": mae,
        "rmse_ohm": rmse,
        "median_abs_diff_ohm": float(np.median(diff_fg)),
        "p95_abs_diff_ohm": float(np.percentile(diff_fg, 95)),
        "max_abs_diff_ohm": float(diff_fg.max()),
        "nrmse_ohm": rmse / r_rng,
        "mape_pct": float(np.mean(diff_fg / np.maximum(real_vals, 1e-9)) * 100.0),
        "real_max_ohm": real_max,
        "gen_max_ohm": gen_max,
        "max_diff_ohm": max_diff_ohm,
        "real_p95_ohm": real_p95,
        "gen_p95_ohm": gen_p95,
        "real_mean_ohm": real_mean,
        "gen_mean_ohm": gen_mean,
        "max_ratio": gen_max / max(real_max, 1e-9),
        "mean_ratio": gen_mean / max(real_mean, 1e-9),
        "pearson_r": pearson_r,
        "spearman_rho": _spearman_rho(real_vals, gen_vals),
        "pattern_mae": float(np.mean(pat_fg)),
        "pattern_rmse": float(np.sqrt(np.mean(pat_fg**2))),
        "peak_loc_err_px": peak_dist,
        "peak_real_ohm": float(real[pr_row, pr_col]),
        "peak_gen_ohm": float(gen[pg_row, pg_col]),
    }
    return row


def _aggregate_by_key(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """Mean numeric metrics grouped by ``key`` (e.g. mhz or k)."""
    groups: dict[Any, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r.get(key), []).append(r)
    out: list[dict[str, Any]] = []
    numeric = [c for c in METRIC_COLUMNS if c not in ("mhz", "k", "sample", "label")]
    for gkey, grp in sorted(groups.items(), key=lambda x: (x[0] is None, x[0])):
        agg: dict[str, Any] = {key: gkey, "n": len(grp)}
        for col in numeric:
            vals = [float(r[col]) for r in grp if r.get(col) is not None and np.isfinite(r[col])]
            agg[f"{col}_mean"] = float(np.mean(vals)) if vals else float("nan")
        out.append(agg)
    return out


def format_metrics_table(rows: list[dict[str, Any]], *, columns: Iterable[str] | None = None) -> str:
    cols = list(columns or METRIC_COLUMNS)
    widths = {c: max(len(c), *(len(f"{r.get(c, ''):.4g}" if isinstance(r.get(c), float) else str(r.get(c, ''))) for r in rows)) for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    sep = "-|-".join("-" * widths[c] for c in cols)
    lines = [header, sep]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, float):
                cells.append(f"{v:.4g}".ljust(widths[c]))
            elif v is None:
                cells.append("".ljust(widths[c]))
            else:
                cells.append(str(v).ljust(widths[c]))
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def write_sim_metrics_bundle(
    rows: list[dict[str, Any]],
    out_root: Path,
    *,
    source: str = "ecadstar_simulated_real",
) -> tuple[Path, Path, Path]:
    """Write CSV + JSON + agent-friendly markdown under ``out_root``."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    csv_path = out_root / SIM_METRICS_CSV
    json_path = out_root / SIM_METRICS_JSON
    md_path = out_root / SIM_METRICS_MD

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(METRIC_COLUMNS), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in METRIC_COLUMNS})

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "description": "Generated heatmap vs ECADStar-simulated real (.map after move)",
        "n_rows": len(rows),
        "per_sample": rows,
        "by_mhz": _aggregate_by_key(rows, "mhz"),
        "by_k": _aggregate_by_key(rows, "k"),
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# Simulated Real vs Generated — Heatmap Metrics",
        "",
        f"- **Rows**: {len(rows)} (one per freq × K × sample)",
        f"- **Source**: ECADStar `.map` in `K*/Real/` vs `data_sample_*/heatmap_physical.npy`",
        f"- **CSV**: `{SIM_METRICS_CSV}`",
        f"- **JSON**: `{SIM_METRICS_JSON}`",
        "",
        "**Primary QC** (tables below): `pearson_r`, `max_diff_ohm` (real−gen; − = gen higher), "
        "`pattern_mae`, `max_ratio`.",
        "Use `mae_ohm` / `mape_pct` from CSV with care — often misleading at 10 MHz (FG Ω ≈ 0).",
        "",
        "## Per-sample metrics (primary)",
        "",
        "```",
        format_metrics_table(rows, columns=PRIMARY_METRIC_COLUMNS),
        "```",
        "",
        "## Mean by MHz (primary)",
        "",
    ]
    by_mhz = payload["by_mhz"]
    if by_mhz:
        md_lines.append("```")
        md_lines.append(format_metrics_table(by_mhz, columns=PRIMARY_BY_MHZ_COLUMNS))
        md_lines.append("```")
    else:
        md_lines.append("(no rows)")

    md_lines.extend([
        "",
        "## Agent copy block",
        "",
        "```",
        f"sim_compare_metrics: n={len(rows)}",
    ])
    for r in rows:
        md_lines.append(
            f"  {r.get('mhz')}MHz K{r.get('k')} s{r.get('sample')}: "
            f"r={r['pearson_r']:.3f} Δmax={r['max_diff_ohm']:+.3g}Ω "
            f"pattern_mae={r['pattern_mae']:.3f} max_ratio={r['max_ratio']:.3f} "
            f"real_max={r['real_max_ohm']:.3g} gen_max={r['gen_max_ohm']:.3g} "
            f"peak_err={r['peak_loc_err_px']:.1f}px"
        )
    md_lines.append("```")
    md_lines.append("")

    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"  metrics → {csv_path.name}, {json_path.name}, {md_path.name}")
    return csv_path, json_path, md_path


def print_metrics_summary(row: dict[str, Any]) -> None:
    """One-line stdout summary for a single comparison (primary metrics first)."""
    dmax = float(row["max_diff_ohm"])
    dnote = "gen>real" if dmax < 0 else ("gen<real" if dmax > 0 else "matched")
    print(
        f"  {row['label']}: r={row['pearson_r']:.3f}  "
        f"Δmax={dmax:+.3g}Ω ({dnote})  "
        f"pattern_mae={row['pattern_mae']:.3f}  "
        f"real_max={row['real_max_ohm']:.3g}  gen_max={row['gen_max_ohm']:.3g}"
    )
