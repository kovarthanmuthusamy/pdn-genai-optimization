#!/usr/bin/env python3
"""Post-process a novelty sweep folder into a more detailed report.

Reads:
- <sweep_root>/novelty_sweep_summary.csv (aggregate medians)
- <sweep_root>/K{K}/novelty_report.csv (per-sample scores)

Writes:
- <sweep_root>/novelty_sweep_summary_detailed.csv
- <sweep_root>/novelty_sweep_summary_detailed.md

This avoids recomputing distances; it summarizes the already-written CSVs.

Additional novelty method (discrete)
- Occupancy-pattern novelty: for each K, measure how often a generated 52-bit
    occupancy pattern was never seen in the training set for that K.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import math
from pathlib import Path
from typing import Any

import numpy as np


def _q(x: list[float]) -> dict[str, float | None]:
    if not x:
        return {"min": None, "p10": None, "median": None, "p90": None, "max": None}
    arr = np.asarray(x, dtype=np.float64)
    qs = np.quantile(arr, [0.0, 0.1, 0.5, 0.9, 1.0])
    return {"min": float(qs[0]), "p10": float(qs[1]), "median": float(qs[2]), "p90": float(qs[3]), "max": float(qs[4])}


def _fmt(v: float | None) -> str:
    return "NA" if v is None else f"{v:.4g}"


def _as_pct(v: float | None) -> str:
    if v is None:
        return "NA"
    return f"{100.0 * float(v):.2f}%"


def _extract_block(text: str, *, begin: str, end: str) -> str | None:
    """Extract a preserved manual block from an existing markdown file."""
    try:
        i0 = text.index(begin) + len(begin)
        i1 = text.index(end, i0)
    except ValueError:
        return None
    return text[i0:i1].strip("\n")


def _to_float_or_nan(v: Any) -> float:
    if v in (None, "", "nan", "None"):
        return float("nan")
    try:
        return float(v)
    except Exception:
        return float("nan")


def _generate_plots(*, rows_by_k: list[dict[str, Any]], out_dir: Path) -> dict[str, str]:
    """Generate PNG plots into out_dir.

    Returns a dict of plot keys -> markdown-relative paths (e.g. plots/foo.png).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    out_dir.mkdir(parents=True, exist_ok=True)

    ks = np.asarray([int(r["K"]) for r in rows_by_k], dtype=np.int32)

    combined_med = np.asarray([_to_float_or_nan(r.get("combined_median")) for r in rows_by_k], dtype=np.float64)
    baseline_med = np.asarray([_to_float_or_nan(r.get("baseline_combined_median")) for r in rows_by_k], dtype=np.float64)
    train_to_gen_med = np.asarray([_to_float_or_nan(r.get("train_to_gen_combined_median")) for r in rows_by_k], dtype=np.float64)
    ratio = np.asarray([_to_float_or_nan(r.get("median_ratio_gen_over_base")) for r in rows_by_k], dtype=np.float64)

    occ_exact = np.asarray([_to_float_or_nan(r.get("occ_exact_rate")) for r in rows_by_k], dtype=np.float64)
    occ_new = np.asarray([_to_float_or_nan(r.get("gen_occ_new_rate")) for r in rows_by_k], dtype=np.float64)
    occ_cov = np.asarray([_to_float_or_nan(r.get("train_occ_covered_rate")) for r in rows_by_k], dtype=np.float64)

    nn_mode = np.asarray([_to_float_or_nan(r.get("nn_mode_frac")) for r in rows_by_k], dtype=np.float64)
    occ_mode = np.asarray([_to_float_or_nan(r.get("gen_occ_mode_frac")) for r in rows_by_k], dtype=np.float64)

    hm_med = np.asarray([_to_float_or_nan(r.get("hm_median")) for r in rows_by_k], dtype=np.float64)
    imp_med = np.asarray([_to_float_or_nan(r.get("imp_median")) for r in rows_by_k], dtype=np.float64)
    ham_med = np.asarray([_to_float_or_nan(r.get("ham_median")) for r in rows_by_k], dtype=np.float64)

    paths: dict[str, str] = {}

    # 1) Distances (log scale to show baseline + gen together)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(ks, combined_med, marker="o", markersize=3, linewidth=1.5, label="gen→train NN (median)")
    ax.plot(ks, train_to_gen_med, marker="o", markersize=3, linewidth=1.5, label="train→gen NN (median)")
    ax.plot(ks, baseline_med, marker="o", markersize=3, linewidth=1.5, label="train→train baseline (median)")
    ax.set_yscale("log")
    ax.set_title("Combined NN distance vs K (log scale)")
    ax.set_xlabel("K")
    ax.set_ylabel("distance (log)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    p = out_dir / "distance_vs_k.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths["distance"] = f"plots/{p.name}"

    # 2) Ratio
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(ks, ratio, marker="o", markersize=3, linewidth=1.5)
    ax.set_title("Median ratio: gen→train / train→train baseline")
    ax.set_xlabel("K")
    ax.set_ylabel("ratio")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    p = out_dir / "ratio_vs_k.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths["ratio"] = f"plots/{p.name}"

    # 3) Occupancy rates
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(ks, occ_exact, marker="o", markersize=3, linewidth=1.5, label="occ_exact_rate")
    ax.plot(ks, occ_new, marker="o", markersize=3, linewidth=1.5, label="gen_occ_new_rate")
    ax.plot(ks, occ_cov, marker="o", markersize=3, linewidth=1.5, label="train_occ_covered_rate")
    ax.set_title("Occupancy novelty / overlap rates vs K")
    ax.set_xlabel("K")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    p = out_dir / "occupancy_rates_vs_k.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths["occ_rates"] = f"plots/{p.name}"

    # 4) Collapse indicators + components
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(11, 7), sharex=True)
    ax0.plot(ks, nn_mode, marker="o", markersize=3, linewidth=1.5, label="nn_mode_frac")
    ax0.plot(ks, occ_mode, marker="o", markersize=3, linewidth=1.5, label="gen_occ_mode_frac")
    ax0.set_title("Collapse indicators vs K")
    ax0.set_ylabel("fraction")
    ax0.set_ylim(-0.05, 1.05)
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="best")

    ax1.plot(ks, hm_med, marker="o", markersize=3, linewidth=1.5, label="heatmap_mse_median")
    ax1.plot(ks, (ham_med / 52.0), marker="o", markersize=3, linewidth=1.5, label="occ_hamming_median/52")
    ax1.plot(ks, imp_med, marker="o", markersize=3, linewidth=1.5, label="impedance_mse_median")
    ax1.set_title("Combined-score components (medians)")
    ax1.set_xlabel("K")
    ax1.set_ylabel("value")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")

    fig.tight_layout()
    p = out_dir / "collapse_and_components_vs_k.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths["collapse_components"] = f"plots/{p.name}"

    return paths


def _occ_key_from_vec(occ_vec: np.ndarray) -> bytes:
    occ_vec = np.asarray(occ_vec).reshape(-1)
    if occ_vec.size != 52:
        raise ValueError(f"Expected occupancy length 52, got {occ_vec.shape}")
    occ_bin = (occ_vec > 0.5).astype(np.uint8, copy=False)
    return np.packbits(occ_bin).tobytes()


def _entropy_bits(counter: Counter[bytes], total: int) -> float:
    if total <= 0 or not counter:
        return 0.0
    p = np.asarray([c / float(total) for c in counter.values()], dtype=np.float64)
    return float(-(p * np.log2(p)).sum())


def _jsd_bits(p_counter: Counter[bytes], p_total: int, q_counter: Counter[bytes], q_total: int) -> float:
    if p_total <= 0 or q_total <= 0:
        return 0.0
    keys = list(set(p_counter.keys()) | set(q_counter.keys()))
    p = np.asarray([p_counter.get(k, 0) / float(p_total) for k in keys], dtype=np.float64)
    q = np.asarray([q_counter.get(k, 0) / float(q_total) for k in keys], dtype=np.float64)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())

    return 0.5 * (_kl(p, m) + _kl(q, m))


def _load_or_build_k_cache(dataset_root: Path) -> tuple[np.ndarray, np.ndarray]:
    cache_path = dataset_root / "k_index_cache.npz"
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        return data["stems"], data["k"]

    occ_dir = dataset_root / "Occ_map"
    if not occ_dir.exists():
        raise SystemExit(f"Missing occupancy directory for K cache: {occ_dir}")

    occ_files = sorted(occ_dir.glob("*.npy"))
    if not occ_files:
        raise SystemExit(f"No occupancy .npy files found in {occ_dir}")

    stems: list[str] = []
    ks: list[int] = []
    for i, occ_path in enumerate(occ_files):
        occ = np.load(occ_path, mmap_mode="r").reshape(-1)
        k = int((occ > 0.5).sum())
        stems.append(occ_path.stem)
        ks.append(k)
        if (i + 1) % 2000 == 0:
            print(f"Scanned {i+1}/{len(occ_files)} occupancy files...")

    stems_arr = np.asarray(stems)
    k_arr = np.asarray(ks, dtype=np.int16)
    np.savez_compressed(cache_path, stems=stems_arr, k=k_arr)
    print(f"Wrote K cache: {cache_path}")
    return stems_arr, k_arr


def _build_train_occ_counters(*, dataset_root: Path, ks: set[int]) -> tuple[dict[int, Counter[bytes]], dict[int, int]]:
    occ_dir = dataset_root / "Occ_map"
    if not occ_dir.exists():
        raise SystemExit(f"Missing dataset occupancy dir: {occ_dir}")

    stems, k_vals = _load_or_build_k_cache(dataset_root)
    counters: dict[int, Counter[bytes]] = {k: Counter() for k in ks}
    totals: dict[int, int] = {k: 0 for k in ks}

    # Iterate once over the dataset and accumulate counts for requested Ks.
    for stem, k in zip(stems.tolist(), k_vals.tolist()):
        kk = int(k)
        if kk not in ks:
            continue
        occ_path = occ_dir / f"{stem}.npy"
        if not occ_path.exists():
            continue
        key = _occ_key_from_vec(np.load(occ_path, mmap_mode="r"))
        counters[kk][key] += 1
        totals[kk] += 1

    return counters, totals


def _load_aggregate_summary(path: Path) -> dict[int, dict[str, Any]]:
    by_k: dict[int, dict[str, Any]] = {}
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            k = int(row["K"])
            by_k[k] = row
    return by_k


def _load_per_k_report(path: Path) -> dict[str, Any]:
    combined: list[float] = []
    hm: list[float] = []
    imp: list[float] = []
    ham: list[float] = []
    nn_percentile: list[float] = []
    exact_count = 0
    nn_samples: list[str] = []

    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            combined.append(float(row["combined_score"]))
            hm.append(float(row["heatmap_mse"]))
            imp.append(float(row["impedance_mse"]))
            ham.append(float(row["occupancy_hamming"]))
            pct = row.get("combined_percentile_vs_train_nn")
            if pct not in (None, "", "nan", "None"):
                try:
                    nn_percentile.append(float(pct))
                except Exception:
                    pass
            if str(row["occupancy_exact"]).lower() == "true":
                exact_count += 1
            nn_samples.append(row["nn_train_sample"])

    n = len(combined)
    nn_mode = ""
    nn_mode_frac = 0.0
    if nn_samples:
        c = Counter(nn_samples)
        nn_mode, nn_mode_count = c.most_common(1)[0]
        nn_mode_frac = float(nn_mode_count) / float(n)

    return {
        "n_gen": n,
        "combined_raw": combined,
        "hm_raw": hm,
        "imp_raw": imp,
        "ham_raw": ham,
        "nn_percentile_raw": nn_percentile,
        "combined": _q(combined),
        "hm": _q(hm),
        "imp": _q(imp),
        "ham": _q(ham),
        "nn_percentile": _q(nn_percentile),
        "occ_exact": exact_count,
        "occ_exact_rate": (float(exact_count) / float(n)) if n else 0.0,
        "nn_mode": nn_mode,
        "nn_mode_frac": nn_mode_frac,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sweep-root",
        type=Path,
        default=Path("evaluation/novelty/runs/novelty_sweep_N100"),
        help="Sweep output root containing novelty_sweep_summary.csv and K*/ folders",
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/data_norm"),
        help="Dataset root (used for occupancy-pattern novelty)",
    )
    ap.add_argument(
        "--no-occ-novelty",
        action="store_true",
        help="Disable occupancy-pattern novelty analysis",
    )
    ap.add_argument(
        "--plots",
        action="store_true",
        help="Generate PNG plots into <sweep_root>/plots and embed them in the markdown",
    )
    args = ap.parse_args()

    sweep_root: Path = args.sweep_root
    agg_path = sweep_root / "novelty_sweep_summary.csv"
    if not agg_path.exists():
        raise SystemExit(f"Missing aggregate summary: {agg_path}")

    agg = _load_aggregate_summary(agg_path)

    ks = sorted(int(p.name[1:]) for p in sweep_root.glob("K*") if p.is_dir() and p.name[1:].isdigit())
    if not ks:
        raise SystemExit(f"No K* folders found under {sweep_root}")

    compute_occ_novelty = not bool(args.no_occ_novelty)
    train_occ_counters: dict[int, Counter[bytes]] = {}
    train_occ_totals: dict[int, int] = {}
    if compute_occ_novelty:
        print("Building training occupancy-pattern counters...")
        train_occ_counters, train_occ_totals = _build_train_occ_counters(dataset_root=args.dataset_root, ks=set(ks))

    dataset_total_occ = sum(int(train_occ_totals.get(k, 0)) for k in ks) if compute_occ_novelty else 0
    dataset_total_occ_unique = sum(len(train_occ_counters.get(k, Counter())) for k in ks) if compute_occ_novelty else 0

    rows: list[dict[str, Any]] = []
    all_exact = 0
    all_gen = 0
    all_combined: list[float] = []
    all_hm: list[float] = []
    all_imp: list[float] = []
    all_ham: list[float] = []
    all_nn_pct: list[float] = []
    all_occ_new = 0
    all_occ_in_train_unique = 0
    all_occ_max_possible_in_train_unique = 0

    # Gather per-K stats
    for k in ks:
        per_path = sweep_root / f"K{k}" / "novelty_report.csv"
        if not per_path.exists():
            continue

        per = _load_per_k_report(per_path)
        all_exact += int(per["occ_exact"])
        all_gen += int(per["n_gen"])
        all_combined.extend(per["combined_raw"])
        all_hm.extend(per["hm_raw"])
        all_imp.extend(per["imp_raw"])
        all_ham.extend(per["ham_raw"])
        all_nn_pct.extend(per.get("nn_percentile_raw", []))

        agg_row = agg.get(k, {})
        n_train = int(float(agg_row.get("n_train", "nan"))) if agg_row.get("n_train") not in (None, "") else None
        base_med = agg_row.get("baseline_combined_median", None)
        base_med = float(base_med) if base_med not in (None, "", "nan") else None

        def _to_float(v: Any) -> float | None:
            if v in (None, "", "nan", "None"):
                return None
            try:
                return float(v)
            except Exception:
                return None

        train_to_gen_med = _to_float(agg_row.get("train_to_gen_combined_median"))
        train_to_gen_cov_p90 = _to_float(agg_row.get("train_to_gen_cov_at_base_p90"))
        mem_p01 = _to_float(agg_row.get("mem_suspect_rate_p01"))
        mem_p05 = _to_float(agg_row.get("mem_suspect_rate_p05"))
        mem_p10 = _to_float(agg_row.get("mem_suspect_rate_p10"))

        gen_med = per["combined"]["median"]
        ratio = (float(gen_med) / float(base_med)) if (base_med not in (None, 0.0) and gen_med is not None) else None

        occ_metrics: dict[str, Any] = {
            "train_occ_unique": None,
            "gen_occ_unique": None,
            "gen_occ_new_unique": None,
            "gen_occ_new_rate": None,
            "gen_occ_in_train_unique": None,
            "train_occ_covered_rate": None,
            "max_possible_train_coverage_rate": None,
            "gen_occ_repeat_rate": None,
            "gen_occ_mode_frac": None,
            "occ_jsd": None,
            "train_occ_entropy": None,
            "gen_occ_entropy": None,
            "combo_space": None,
            "combo_space_log10": None,
            "train_combo_space_rate": None,
            "gen_combo_space_rate": None,
            "gen_in_train_combo_space_rate": None,
            "gen_occ_in_train_samples": None,
            "gen_occ_in_train_sample_rate": None,
            "expected_in_train_samples_uniform": None,
            "expected_in_train_sample_rate_uniform": None,
            "in_train_sample_rate_over_uniform": None,
        }
        if compute_occ_novelty:
            gen_occ_path = sweep_root / f"K{k}" / "occupancy.npy"
            if gen_occ_path.exists():
                gen_occ = np.load(gen_occ_path)
                gen_occ = np.asarray(gen_occ).reshape(-1, 52)
                gen_keys = [_occ_key_from_vec(v) for v in gen_occ]
                gen_counter = Counter(gen_keys)
                n_gen = int(per["n_gen"])
                gen_unique = len(gen_counter)
                gen_mode_frac = (max(gen_counter.values()) / float(n_gen)) if (n_gen > 0 and gen_counter) else 0.0

                train_counter = train_occ_counters.get(k, Counter())
                train_total = int(train_occ_totals.get(k, 0))
                train_unique = len(train_counter)

                # Upper bound on unique dataset patterns we can possibly hit for this K
                # given we only generated n_gen samples.
                all_occ_max_possible_in_train_unique += int(min(int(n_gen), int(train_unique)))

                train_set = set(train_counter.keys())
                gen_new_unique = len(set(gen_counter.keys()) - train_set)
                gen_in_train_unique = len(set(gen_counter.keys()) & train_set)
                gen_new_count = sum(c for key, c in gen_counter.items() if key not in train_set)
                gen_new_rate = (gen_new_count / float(n_gen)) if n_gen > 0 else 0.0
                all_occ_new += int(gen_new_count)
                all_occ_in_train_unique += int(gen_in_train_unique)

                gen_in_train_samples = int(n_gen) - int(gen_new_count)
                gen_in_train_sample_rate = (gen_in_train_samples / float(n_gen)) if n_gen > 0 else 0.0

                train_covered_rate = (gen_in_train_unique / float(train_unique)) if train_unique > 0 else 0.0
                max_possible_coverage_rate = (min(int(n_gen), int(train_unique)) / float(train_unique)) if train_unique > 0 else 0.0
                gen_repeat_rate = 1.0 - (float(gen_unique) / float(n_gen)) if n_gen > 0 else 0.0

                try:
                    combo_space = math.comb(52, int(k))
                    combo_space_log10 = float(math.log10(combo_space)) if combo_space > 0 else 0.0
                except Exception:
                    combo_space = None
                    combo_space_log10 = None

                train_combo_space_rate = (float(train_unique) / float(combo_space)) if (combo_space not in (None, 0)) else None
                gen_combo_space_rate = (float(gen_unique) / float(combo_space)) if (combo_space not in (None, 0)) else None
                gen_in_train_combo_space_rate = (float(gen_in_train_unique) / float(combo_space)) if (combo_space not in (None, 0)) else None

                expected_in_train_sample_rate_uniform = train_combo_space_rate
                expected_in_train_samples_uniform = (float(n_gen) * float(expected_in_train_sample_rate_uniform)) if expected_in_train_sample_rate_uniform is not None else None
                in_train_sample_rate_over_uniform = (
                    (float(gen_in_train_sample_rate) / float(expected_in_train_sample_rate_uniform))
                    if (expected_in_train_sample_rate_uniform not in (None, 0.0))
                    else None
                )

                occ_metrics.update(
                    {
                        "train_occ_unique": int(train_unique),
                        "gen_occ_unique": int(gen_unique),
                        "gen_occ_new_unique": int(gen_new_unique),
                        "gen_occ_new_rate": float(gen_new_rate),
                        "gen_occ_in_train_unique": int(gen_in_train_unique),
                        "train_occ_covered_rate": float(train_covered_rate),
                        "max_possible_train_coverage_rate": float(max_possible_coverage_rate),
                        "gen_occ_repeat_rate": float(gen_repeat_rate),
                        "gen_occ_mode_frac": float(gen_mode_frac),
                        "occ_jsd": float(_jsd_bits(train_counter, train_total, gen_counter, n_gen)),
                        "train_occ_entropy": float(_entropy_bits(train_counter, train_total)),
                        "gen_occ_entropy": float(_entropy_bits(gen_counter, n_gen)),
                        "combo_space": combo_space,
                        "combo_space_log10": combo_space_log10,
                        "train_combo_space_rate": train_combo_space_rate,
                        "gen_combo_space_rate": gen_combo_space_rate,
                        "gen_in_train_combo_space_rate": gen_in_train_combo_space_rate,
                        "gen_occ_in_train_samples": int(gen_in_train_samples),
                        "gen_occ_in_train_sample_rate": float(gen_in_train_sample_rate),
                        "expected_in_train_samples_uniform": expected_in_train_samples_uniform,
                        "expected_in_train_sample_rate_uniform": expected_in_train_sample_rate_uniform,
                        "in_train_sample_rate_over_uniform": in_train_sample_rate_over_uniform,
                    }
                )

        rows.append(
            {
                "K": k,
                "n_train": n_train,
                "n_gen": per["n_gen"],
                "baseline_combined_median": base_med,
                "combined_p10": per["combined"]["p10"],
                "combined_median": per["combined"]["median"],
                "combined_p90": per["combined"]["p90"],
                "median_ratio_gen_over_base": ratio,
                "hm_median": per["hm"]["median"],
                "imp_median": per["imp"]["median"],
                "ham_median": per["ham"]["median"],
                "occ_exact": per["occ_exact"],
                "occ_exact_rate": per["occ_exact_rate"],
                "nn_mode_frac": per["nn_mode_frac"],
                "nn_mode_sample": per["nn_mode"],
                "gen_nn_percentile_p10": per["nn_percentile"]["p10"],
                "gen_nn_percentile_median": per["nn_percentile"]["median"],
                "gen_nn_percentile_p90": per["nn_percentile"]["p90"],
                "mem_suspect_rate_p01": mem_p01,
                "mem_suspect_rate_p05": mem_p05,
                "mem_suspect_rate_p10": mem_p10,
                "train_to_gen_combined_median": train_to_gen_med,
                "train_to_gen_cov_at_base_p90": train_to_gen_cov_p90,
                **occ_metrics,
            }
        )

    if not rows:
        raise SystemExit("No per-K novelty_report.csv files found")

    # Write detailed CSV
    out_csv = sweep_root / "novelty_sweep_summary_detailed.csv"
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Build markdown
    out_md = sweep_root / "novelty_sweep_summary_detailed.md"

    # Keep one stable ordering for summaries/plots.
    rows_by_k = sorted(rows, key=lambda r: int(r["K"]))

    global_q = {
        "combined": _q(all_combined),
        "hm": _q(all_hm),
        "imp": _q(all_imp),
        "ham": _q(all_ham),
        "nn_percentile": _q(all_nn_pct),
    }

    # Aggregate / sanity helpers
    overall_occ_exact_rate = (float(all_exact) / float(all_gen)) if all_gen else 0.0
    overall_occ_new_rate = (float(all_occ_new) / float(all_gen)) if (compute_occ_novelty and all_gen) else None
    overall_unique_train_cov_rate = (
        (float(all_occ_in_train_unique) / float(dataset_total_occ_unique))
        if (compute_occ_novelty and dataset_total_occ_unique > 0)
        else None
    )

    # Pull optional K-level fields from rows (these exist only if the sweep CSV has them)
    k_mem_meds = [r.get("gen_nn_percentile_median") for r in rows if r.get("gen_nn_percentile_median") is not None]
    k_cov = [r.get("train_to_gen_cov_at_base_p90") for r in rows if r.get("train_to_gen_cov_at_base_p90") is not None]
    all_pct_saturated_one = bool(all_nn_pct) and (np.nanmin(np.asarray(all_nn_pct)) >= 1.0) and (np.nanmax(np.asarray(all_nn_pct)) <= 1.0)
    all_cov_zero = bool(k_cov) and all(float(v) == 0.0 for v in k_cov if v is not None)

    # Preserve a human-written summary block across regenerations
    manual_begin = "<!-- BEGIN HUMAN SUMMARY -->"
    manual_end = "<!-- END HUMAN SUMMARY -->"
    manual_summary = ""
    if out_md.exists():
        prev = out_md.read_text(encoding="utf-8")
        manual_summary = _extract_block(prev, begin=manual_begin, end=manual_end) or ""
    if not manual_summary.strip():
        manual_summary = "(Write your human summary here; this block is preserved on regeneration.)"

    plot_paths: dict[str, str] = {}
    if bool(args.plots):
        plot_paths = _generate_plots(rows_by_k=rows_by_k, out_dir=sweep_root / "plots")

    ratio_vals: list[float] = []
    for r in rows_by_k:
        rv = r.get("median_ratio_gen_over_base")
        if isinstance(rv, (int, float)) and not np.isnan(float(rv)):
            ratio_vals.append(float(rv))
    ratio_min = min(ratio_vals) if ratio_vals else None
    ratio_max = max(ratio_vals) if ratio_vals else None

    md: list[str] = []
    md.append("# VAE novelty sweep — summary\n")
    md.append(f"Sweep root: `{sweep_root}`\n")

    md.append("## At a glance\n")
    md.append(f"- Total generated: **{all_gen}** samples across **{len(rows_by_k)}** K values")
    md.append(f"- Occupancy exact-match rate: **{_as_pct(overall_occ_exact_rate)}**")
    md.append(f"- Combined_score (median): **{_fmt(global_q['combined']['median'])}**")
    if ratio_min is not None and ratio_max is not None:
        md.append(f"- gen/base median-distance ratio: **{_fmt(ratio_min)}×–{_fmt(ratio_max)}×**")
    if compute_occ_novelty:
        md.append(f"- NEW occupancy patterns (rate): **{_as_pct(overall_occ_new_rate)}**")
        md.append(f"- Training occupancy coverage (union): **{_as_pct(overall_unique_train_cov_rate)}**")
    if global_q["nn_percentile"]["median"] is not None:
        md.append(f"- gen→train NN percentile (median): **{_fmt(global_q['nn_percentile']['median'])}**")
    md.append("")

    md.append("## Key points\n")

    if compute_occ_novelty:
        ks_no_overlap: list[int] = []
        for r in rows_by_k:
            in_train_unique = r.get("gen_occ_in_train_unique")
            if isinstance(in_train_unique, (int, float)) and int(in_train_unique) == 0:
                ks_no_overlap.append(int(r["K"]))

        if ks_no_overlap:
            ks_no_overlap.sort()
            if ks_no_overlap[-1] - ks_no_overlap[0] + 1 == len(ks_no_overlap):
                md.append(
                    f"- Mid-range K={ks_no_overlap[0]}..{ks_no_overlap[-1]} shows **0** occupancy overlap with training (exact-set)."
                )
            else:
                md.append("- Many K values show **0** occupancy overlap with training (exact-set).")

        if overall_occ_new_rate is not None:
            md.append("- Occupancy novelty is high overall (see “At a glance”).")
        if overall_unique_train_cov_rate is not None:
            md.append("- Training occupancy coverage remains low (see “At a glance”).")

    if all_pct_saturated_one:
        md.append(
            "- gen→train NN percentiles saturate near 1.0 (no direct-copy signal under this metric, but the combined distance scale is strict)."
        )
    elif global_q["nn_percentile"]["median"] is not None:
        md.append(
            "- gen→train NN percentile (median) is "
            f"{_fmt(global_q['nn_percentile']['median'])} (lower would be more memorization-like)."
        )

    if all_cov_zero:
        md.append(
            "- train→gen coverage@baseline p90 is 0 for all K under the current combined metric (generated samples are not within a typical train→train neighborhood)."
        )

    md.append("- `combined_score` is dominated by heatmap MSE (other terms are much smaller).")

    degenerate_ks = sorted({int(r["K"]) for r in rows_by_k if int(r.get("n_train") or 0) <= 1})
    if degenerate_ks:
        md.append(
            "- Some extreme K values have very small training support (e.g., "
            f"K={degenerate_ks[0]}), so overlap/collapse metrics there are not representative."
        )

    md.append("")

    if plot_paths:
        md.append("## Plots\n")
        md.append("(Images are saved under `plots/` relative to the sweep root.)\n")

        md.append("### Distances vs K\n")
        md.append(f"![Combined NN distance vs K]({plot_paths['distance']})\n")
        md.append("Shows gen→train NN distances vs the train→train NN baseline (log scale).\n")

        md.append("### Gen/Base ratio vs K\n")
        md.append(f"![Gen/Base ratio vs K]({plot_paths['ratio']})\n")
        md.append("`median_ratio_gen_over_base` = gen_median / baseline_median (higher = further from training in this metric).\n")

        md.append("### Occupancy novelty/overlap vs K\n")
        md.append(f"![Occupancy rates vs K]({plot_paths['occ_rates']})\n")
        md.append("Exact-set rates computed from occupancy patterns only.\n")

        md.append("### Collapse indicators + components\n")
        md.append(f"![Collapse and components vs K]({plot_paths['collapse_components']})\n")
        md.append("Top: mode shares (collapse indicators). Bottom: median components contributing to `combined_score`.\n")

    md.append("## Human summary (manual)\n")
    md.append(manual_begin)
    md.append(manual_summary)
    md.append(manual_end)
    md.append("")

    md.append("## Files\n")
    md.append(f"- Per-K summary CSV (all metrics): `{out_csv}`")
    md.append(f"- Sweep aggregate CSV: `{agg_path}`")
    if plot_paths:
        md.append(f"- Plots folder: `{sweep_root / 'plots'}`")

    md.append("\n## Notes\n")
    md.append("- This markdown is intentionally compact; use the CSV for full per-K tables and additional metrics.")
    md.append("- `combined_score = heatmap_mse + impedance_mse + (occ_hamming / 52)`.")
    md.append("- Extreme K values can have very small training support, so overlap/collapse metrics there can be misleading.")
    if all_pct_saturated_one or all_cov_zero:
        md.append(
            "- If percentile/coverage metrics saturate, consider rescaling distances (e.g., normalize each modality by its train→train baseline scale)."
        )

    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print("✓ Wrote")
    print(f"  {out_csv}")
    print(f"  {out_md}")


if __name__ == "__main__":
    main()
