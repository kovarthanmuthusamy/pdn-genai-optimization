"""Off-anchor eval hook for exp055 training checkpoints."""

from __future__ import annotations

from pathlib import Path

from experiments.exp059_capacity_freq.codes.eval_spatial_metrics import (
    OFF_ANCHOR_CSV_NAME,
    epoch_from_off_anchor_path,
    run_off_anchor_eval_spatial,
)

_OFF_ANCHOR_CFG: list = []


def set_off_anchor_config(c) -> None:
    _OFF_ANCHOR_CFG.clear()
    _OFF_ANCHOR_CFG.append(c)



def _interval(c) -> int:
    if c is None:
        return 50
    v = getattr(c, "eval_off_anchor_interval", 50)
    try:
        iv = int(v) if v is not None else 50
    except (TypeError, ValueError):
        iv = 50
    return max(0, iv)


def should_run_off_anchor(epoch: int, c) -> bool:
    interval = _interval(c)
    if interval <= 0:
        return False
    return epoch > 0 and epoch % interval == 0



def run_off_anchor_eval(
    model,
    val_loader,
    *,
    bg: float,
    off_anchor_mhz=(100.0, 270.0, 400.0),
    max_batches: int = 12,
    device: str = "cuda",
    out_csv=None,
):
    epoch = epoch_from_off_anchor_path(out_csv)
    c = _OFF_ANCHOR_CFG[0] if _OFF_ANCHOR_CFG else None
    if not should_run_off_anchor(epoch, c):
        return []
    fixed = Path(out_csv).parent / OFF_ANCHOR_CSV_NAME if out_csv is not None else None
    use_binary_occ = bool(getattr(c, "occupancy_binary_decode", True)) if c else True
    use_occ_only = bool(getattr(c, "eval_use_occ_only_layout", False)) if c else False
    return run_off_anchor_eval_spatial(
        model,
        val_loader,
        bg=bg,
        off_anchor_mhz=off_anchor_mhz,
        max_batches=max_batches,
        device=device,
        out_csv=fixed,
        epoch=epoch,
        use_binary_occupancy=use_binary_occ,
        use_occ_only_layout=use_occ_only,
    )


def off_anchor_aggregate_score(
    rows: list[dict],
    *,
    kind: str = "layout_cross",
    weights: dict | None = None,
) -> float:
    """Weighted mean FG MSE across off-anchor MHz (lower = better heatmap fit)."""
    scored = [r for r in rows if r.get("kind") == kind and r.get("hm_fg_mse_mean") is not None]
    if not scored:
        return float("inf")
    wmap = {float(k): float(v) for k, v in (weights or {}).items()}
    total = 0.0
    total_w = 0.0
    for r in scored:
        mhz = float(r["mhz"])
        w = wmap.get(mhz, 1.0)
        total += w * float(r["hm_fg_mse_mean"])
        total_w += w
    return total / total_w if total_w else float("inf")
