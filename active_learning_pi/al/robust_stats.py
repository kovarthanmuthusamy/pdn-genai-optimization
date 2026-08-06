"""Robust foreground peak statistics on physical heatmaps.

Run:
    Import only — used by ``inference_pool`` and ``evaluate_off_anchor``."""
from __future__ import annotations

import numpy as np


def _peak_location_2d(arr: np.ndarray, mask: np.ndarray | None = None) -> tuple[int, int]:
    """Argmax (row, col) within mask; ties → first occurrence."""
    masked = np.where(mask > 0.5, arr, -np.inf) if mask is not None else arr
    flat_idx = int(np.argmax(masked))
    return divmod(flat_idx, arr.shape[1])


def foreground_mask(hm: np.ndarray, bg_margin: float = 0.5) -> np.ndarray:
    """hm: (2,H,W) z-score or (H,W) single channel — use channel 0."""
    if hm.ndim == 3:
        ch = hm[0]
    else:
        ch = hm
    return ch > bg_margin


def robust_peak_stats(hm_phys: np.ndarray, mask_board: np.ndarray | None = None) -> dict[str, float]:
    """
    hm_phys: (2,H,W) physical impedance × mask or (H,W).
    Returns foreground p95, p99, top-k mean (k=1% of fg pixels).
    """
    if hm_phys.ndim == 3:
        z = hm_phys[0] * hm_phys[1] if hm_phys.shape[0] >= 2 else hm_phys[0]
    else:
        z = hm_phys
    fg = z > 1e-6
    if mask_board is not None:
        fg = fg & (mask_board > 0.5)
    vals = z[fg]
    if vals.size == 0:
        return {
            "p95": 0.0,
            "p99": 0.0,
            "topk_mean": 0.0,
            "fg_pixels": 0,
            "peak_row": 0.0,
            "peak_col": 0.0,
            "peak_ohm": 0.0,
        }
    k = max(1, int(0.01 * vals.size))
    topk = np.partition(vals, -k)[-k:]
    board_mask = mask_board if mask_board is not None else np.ones_like(z, dtype=bool)
    peak_row, peak_col = _peak_location_2d(z, board_mask)
    return {
        "p95": float(np.percentile(vals, 95)),
        "p99": float(np.percentile(vals, 99)),
        "topk_mean": float(topk.mean()),
        "fg_pixels": int(vals.size),
        "peak_row": float(peak_row),
        "peak_col": float(peak_col),
        "peak_ohm": float(z[peak_row, peak_col]),
    }


def metric_from_stats(stats_list: list[dict[str, float]], key: str = "p99") -> float:
    arr = np.array([s[key] for s in stats_list], dtype=np.float64)
    return float(arr.var()) if arr.size > 1 else 0.0


def peak_loc_spread(stats_list: list[dict[str, float]]) -> float:
    """RMS spread (px) of argmax peak location across MC passes."""
    if len(stats_list) < 2:
        return 0.0
    rows = np.array([s["peak_row"] for s in stats_list], dtype=np.float64)
    cols = np.array([s["peak_col"] for s in stats_list], dtype=np.float64)
    return float(np.sqrt(rows.var() + cols.var()))


def combined_uncertainty(
    stats_list: list[dict[str, float]],
    *,
    magnitude_key: str = "p99",
    spatial_weight: float = 1.0,
) -> tuple[float, float, float]:
    """Return (total, magnitude_var, peak_loc_spread_px)."""
    mag_var = metric_from_stats(stats_list, magnitude_key)
    spatial = peak_loc_spread(stats_list)
    total = mag_var + float(spatial_weight) * spatial
    return total, mag_var, spatial


def _heatmap_board_flat(hm_phys: np.ndarray, mask_board: np.ndarray | None) -> np.ndarray:
    """Flatten all heatmap channels over the board mask for whole-map comparison."""
    hm = hm_phys[np.newaxis, ...] if hm_phys.ndim == 2 else hm_phys
    if mask_board is not None:
        m = mask_board > 0.5
        return np.stack([hm[c][m] for c in range(hm.shape[0])], axis=0)
    return hm.reshape(hm.shape[0], -1)


def heatmap_mc_sample_mse(
    heatmaps: list[np.ndarray],
    mask_board: np.ndarray | None = None,
) -> float:
    """
    Mean pairwise MSE across MC heatmap samples on the board region (all channels).

    Higher values mean the model disagrees more about the full spatial heatmap,
    not just peak statistics.
    """
    if len(heatmaps) < 2:
        return 0.0
    flats = [_heatmap_board_flat(h, mask_board).astype(np.float64) for h in heatmaps]
    total = 0.0
    count = 0
    for i in range(len(flats)):
        for j in range(i + 1, len(flats)):
            diff = flats[i] - flats[j]
            total += float(np.mean(diff * diff))
            count += 1
    return total / count if count else 0.0


def heatmap_mc_rce(
    heatmaps: list[np.ndarray],
    mask_board: np.ndarray | None = None,
) -> float:
    """Reconstruction consistency error: mean MSE of each sample vs the MC mean."""
    if len(heatmaps) < 1:
        return 0.0
    if len(heatmaps) == 1:
        return 0.0
    flats = [_heatmap_board_flat(h, mask_board).astype(np.float64) for h in heatmaps]
    mean = np.mean(np.stack(flats, axis=0), axis=0)
    return float(np.mean([(f - mean) ** 2 for f in flats]))


def mc_p99_range(stats_list: list[dict[str, float]]) -> float:
    """Peak p99 spread across MC decoder passes (MHz-agnostic disagreement)."""
    if len(stats_list) < 2:
        return 0.0
    arr = np.array([s["p99"] for s in stats_list], dtype=np.float64)
    return float(arr.max() - arr.min())


def relative_prior_deficit(
    pred_p99: float,
    mhz: float,
    fg_max_table: dict[float, float],
    *,
    interp_fn=None,
) -> float:
    """
    Unitless shortfall vs training-set fg-max prior at ``mhz`` (0 = at/above prior).

    Uses the same formula at every frequency — no per-MHz weights or allowlists.
    """
    if pred_p99 <= 0 or not fg_max_table:
        return 0.0
    anchors = sorted(float(k) for k in fg_max_table.keys())
    if not anchors:
        return 0.0
    if interp_fn is not None:
        target = float(interp_fn(float(mhz), fg_max_table))
    else:
        from experiments.exp038_true_multi.codes.freq_inference_utils import bracket_anchors_mhz

        lo, hi, t = bracket_anchors_mhz(float(mhz), anchors)
        v_lo = float(fg_max_table.get(lo, fg_max_table.get(float(lo), 0.0)))
        v_hi = float(fg_max_table.get(hi, fg_max_table.get(float(hi), v_lo)))
        target = v_lo if lo == hi else (1.0 - t) * v_lo + t * v_hi
    if target <= 0:
        return 0.0
    return float(max(0.0, 1.0 - float(pred_p99) / target))


def peak_amplitude_bias_score(
    pred_p99: float,
    mhz: float,
    fg_max_table: dict[float, float],
    *,
    mhz_weights: dict[float, float] | None = None,
    interp_fn=None,
) -> float:
    """Legacy squared deficit — prefer ``auto_acquire_badness`` for new runs."""
    deficit = relative_prior_deficit(pred_p99, mhz, fg_max_table, interp_fn=interp_fn)
    if deficit <= 0:
        return 0.0
    w = 1.0
    if mhz_weights:
        w = float(mhz_weights.get(float(mhz), mhz_weights.get(str(int(mhz)), 1.0)))
    return w * deficit * deficit


def auto_acquire_badness(
    *,
    decoder_rce: float,
    latent_mse: float,
    p99_var: float,
    spatial_spread: float,
    p99_range: float,
    pred_p99_mean: float,
    pred_p99_std: float,
    prior_deficit: float,
    weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """
    MHz-agnostic acquisition badness from model self-disagreement only.

    Epistemic terms (decoder RCE, latent MSE, p99 variance, spatial spread) drive
    ranking. Training-set prior shortfall only **amplifies** uncertain cases — it
    never dominates alone (fixes iter-14 moderate-250 false positives).
    """
    epistemic = (
        weights["decoder_dropout_rce"] * float(decoder_rce)
        + weights["latent_heatmap_mse"] * float(latent_mse)
    )
    disagreement = (
        weights["p99_var"] * float(p99_var)
        + weights["peak_loc_spread"] * float(spatial_spread)
        + weights.get("p99_range", 0.5) * float(p99_range)
    )
    cv = float(pred_p99_std) / max(float(pred_p99_mean), 1e-6)
    cv_term = weights.get("peak_cv", 0.25) * cv

    confusion = epistemic + disagreement + cv_term
    prior_tension = float(prior_deficit) * confusion
    total = confusion + weights.get("prior_tension", 0.35) * prior_tension

    return total, {
        "epistemic_score": epistemic,
        "disagreement_score": disagreement,
        "peak_cv": cv,
        "prior_deficit": float(prior_deficit),
        "prior_tension_score": prior_tension,
    }
