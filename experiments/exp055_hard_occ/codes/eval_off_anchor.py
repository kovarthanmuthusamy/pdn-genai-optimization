"""Off-anchor eval hook for exp055 training checkpoints."""

from __future__ import annotations

from pathlib import Path

from experiments.exp055_hard_occ.codes.eval_spatial_metrics import (
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
    use_encode_skips = bool(getattr(c, "cross_freq_use_encode_skips", False)) if c else False
    use_binary_occ = bool(getattr(c, "occupancy_binary_decode", True)) if c else True
    return run_off_anchor_eval_spatial(
        model,
        val_loader,
        bg=bg,
        off_anchor_mhz=off_anchor_mhz,
        max_batches=max_batches,
        device=device,
        out_csv=fixed,
        epoch=epoch,
        use_encode_skips=use_encode_skips,
        use_binary_occupancy=use_binary_occ,
    )
