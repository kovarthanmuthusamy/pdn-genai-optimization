"""Evaluate native vs cross-frequency heatmap reconstruction on val set.

Purpose:
    Measure foreground MSE when decoding at native MHz vs cross-anchor MHz (layout-z diagnostic).

Run:
    python experiments/exp038_true_multi/codes/eval_cross_freq.py

Agent notes:
    - What: Cross-frequency generalization eval for exp038 VAE heatmap decoder.
    - Usage: Set ``CHECKPOINT``, ``OFF_ANCHOR_MHZ``, ``MAX_BATCHES`` → run. Writes CSV metrics.
    - Config keys:
        - ``CHECKPOINT`` — ``.pt`` path (default ``checkpoints/last_model.pt``)
        - ``MAX_BATCHES`` — val batches to score; ``0`` = full val set
        - ``OUTPUT_CSV`` — where to write per-MHz MSE rows
        - ``OFF_ANCHOR_MHZ`` — MHz list for off-anchor decode test
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve()
PROJECT_ROOT = next(
    (p for p in _ROOT.parents if (p / "src_vae").is_dir() and (p / "experiments").is_dir()),
    _ROOT.parents[3],
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp038_true_multi.codes.dataloader_multifreq import (  # noqa: E402
    ANCHOR_MHZ,
    create_multifreq_data_loaders,
)
from experiments.exp038_true_multi.codes.inference_vae import VAEInference  # noqa: E402
from src_vae.others.pi_freq_utils import pi_freq_mhz_to_norm  # noqa: E402

EXP_DIR = PROJECT_ROOT / "experiments/exp038_true_multi"
DEFAULT_CKPT = EXP_DIR / "checkpoints/last_model.pt"
OUT_CSV = EXP_DIR / "metrics/cross_freq_eval.csv"
DEFAULT_OFF_ANCHOR_MHZ = (80.0, 250.0)

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================

CHECKPOINT = DEFAULT_CKPT
MAX_BATCHES = 0  # 0 = all
OUTPUT_CSV = OUT_CSV
OFF_ANCHOR_MHZ = list(DEFAULT_OFF_ANCHOR_MHZ)

# =============================================================================

def _fg_mse(recon: torch.Tensor, target: torch.Tensor, bg: float, margin: float = 0.5) -> torch.Tensor:
    fg = (target > bg + margin).float()
    err = (recon - target).pow(2)
    n = fg.sum(dim=(1, 2, 3)).clamp(min=1.0)
    return (err * fg).sum(dim=(1, 2, 3)) / n


@torch.inference_mode()
def run_off_anchor_eval(
    model: torch.nn.Module,
    val_loader,
    *,
    bg: float,
    off_anchor_mhz: tuple[float, ...] = DEFAULT_OFF_ANCHOR_MHZ,
    max_batches: int = 30,
    device: str | torch.device = "cuda",
    out_csv: Path | str | None = None,
) -> list[dict]:
    """Layout-z decode at off-anchor MHz; compare to native GT (cross-freq diagnostic)."""
    model.eval()
    base = getattr(model, "_orig_mod", model)
    rows: list[dict] = []
    buckets: dict[tuple[str, float], list[float]] = {}

    for bi, batch in enumerate(val_loader):
        if max_batches > 0 and bi >= max_batches:
            break
        hm = batch["heatmap_norm"].to(device)
        occ = batch["occupancy"].to(device)
        imp = batch["impedance"].to(device)
        K = batch["K"].to(device)
        pi_native = batch["PI_freq"].to(device)
        if hm.dim() == 3:
            hm = hm.unsqueeze(1)
        if imp.dim() == 2:
            imp = imp.unsqueeze(1)
        elif imp.dim() == 3:
            imp = imp[:, :1]
        hm_enc = hm.masked_fill(hm < bg, 0.0)

        z_layout = base.encode_layout_latent(occ, imp, K, pi_native)

        for mhz in off_anchor_mhz:
            pi_t = torch.full((hm.shape[0],), pi_freq_mhz_to_norm(mhz), device=device)
            rh_enc, _, _, _, _, _ = base(hm_enc, occ, imp, K, pi_t)
            rh_lay, _, _ = base.decode(z_layout, K, pi_t)
            mse_enc = _fg_mse(rh_enc, hm, bg)
            mse_lay = _fg_mse(rh_lay, hm, bg)
            for j in range(mse_enc.shape[0]):
                buckets.setdefault(("encode_cross", float(mhz)), []).append(float(mse_enc[j].item()))
                buckets.setdefault(("layout_cross", float(mhz)), []).append(float(mse_lay[j].item()))

    for (kind, mhz), vals in sorted(buckets.items()):
        if not vals:
            continue
        rows.append({
            "mhz": mhz,
            "kind": kind,
            "n": len(vals),
            "hm_fg_mse_mean": sum(vals) / len(vals),
        })

    if out_csv is not None:
        path = Path(out_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["mhz", "kind", "n", "hm_fg_mse_mean"])
            w.writeheader()
            w.writerows(rows)
        print(f"  Off-anchor eval → {path}")

    for r in rows:
        print(
            f"    {r['kind']:14s} @ {r['mhz']:5.0f} MHz  "
            f"MSE={r['hm_fg_mse_mean']:.4f}  n={r['n']}",
        )
    return rows


@torch.inference_mode()
def main() -> None:
    cfg_path = EXP_DIR / "config.yaml"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    bg = float(cfg.get("background_value", -2.8751))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = VAEInference(checkpoint_path=str(CHECKPOINT), device=device)
    stats_auto = EXP_DIR / "metrics/latent_stats.json"
    engine.load_latent_stats(str(stats_auto) if stats_auto.is_file() else None)

    _, val_ld = create_multifreq_data_loaders(
        data_dir=cfg.get("data_dir", str(PROJECT_ROOT / "datasets/data_multifreq_norm")),
        batch_size=int(cfg.get("batch_size", 64)),
        num_workers=2,
        train_split=float(cfg.get("train_split", 0.9)),
        seed=42,
        pin_memory=device.type == "cuda",
        split_by_design=cfg.get("split_by_design", True),
        stratify_by_k=cfg.get("stratify_by_k", True),
        balance_k=False,
        balance_freq=False,
        cache_in_ram=cfg.get("cache_in_ram", True),
        cross_freq_pairs=False,
    )

    native_by_anchor: dict[float, list[float]] = {a: [] for a in ANCHOR_MHZ}
    cross_at: dict[float, list[float]] = {float(m): [] for m in ANCHOR_MHZ}
    model = engine.model
    model.eval()
    base = getattr(model, "_orig_mod", model)

    for bi, batch in enumerate(val_ld):
        if MAX_BATCHES > 0 and bi >= MAX_BATCHES:
            break
        hm = batch["heatmap_norm"].to(device)
        occ = batch["occupancy"].to(device)
        imp = batch["impedance"].to(device)
        K = batch["K"].to(device)
        pi_native = batch["PI_freq"].to(device)
        if hm.dim() == 3:
            hm = hm.unsqueeze(1)
        if imp.dim() == 2:
            imp = imp.unsqueeze(1)
        elif imp.dim() == 3:
            imp = imp[:, :1]
        hm_enc = hm.masked_fill(hm < bg + 0.5, 0.0)

        rh_native, _, _, mu, lv, _ = base(hm_enc, occ, imp, K, pi_native)
        z = base._reparameterize(mu, torch.clamp(lv, -4.0, 2.0))
        mse_n = _fg_mse(rh_native, hm, bg)
        for j in range(mse_n.shape[0]):
            norm_j = float(pi_native[j].item())
            log10_hz = norm_j * 2.778151250383644 + 6.0
            mhz_j = 10 ** log10_hz / 1e6
            anchor = min(ANCHOR_MHZ, key=lambda a: abs(a - mhz_j))
            native_by_anchor[anchor].append(float(mse_n[j].item()))

        for mhz in ANCHOR_MHZ:
            pi_t = torch.full((hm.shape[0],), pi_freq_mhz_to_norm(mhz), device=device)
            rh_cross, _, _ = base.decode(z, K, pi_t)
            cross_at[float(mhz)].extend(_fg_mse(rh_cross, hm, bg).cpu().tolist())

    rows = []
    for anchor in ANCHOR_MHZ:
        nat = native_by_anchor[anchor]
        crs = cross_at[float(anchor)]
        if nat:
            rows.append({
                "mhz": anchor,
                "kind": "native_encode",
                "n": len(nat),
                "hm_fg_mse_mean": sum(nat) / len(nat),
            })
        if crs:
            rows.append({
                "mhz": anchor,
                "kind": "cross_freq_decode",
                "n": len(crs),
                "hm_fg_mse_mean": sum(crs) / len(crs),
            })

    off_anchor = tuple(OFF_ANCHOR_MHZ) if OFF_ANCHOR_MHZ else DEFAULT_OFF_ANCHOR_MHZ
    if off_anchor:
        print(f"\nOff-anchor MHz: {list(off_anchor)}")
        rows.extend(
            run_off_anchor_eval(
                model,
                val_ld,
                bg=bg + 0.5,
                off_anchor_mhz=off_anchor,
                max_batches=MAX_BATCHES or 30,
                device=device,
                out_csv=None,
            ),
        )

    out = Path(OUTPUT_CSV)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["mhz", "kind", "n", "hm_fg_mse_mean"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}")
    for r in rows:
        print(f"  {r['kind']:20s} @ {r['mhz']:3.0f} MHz  MSE={r['hm_fg_mse_mean']:.4f}  n={r['n']}")


if __name__ == "__main__":
    main()
