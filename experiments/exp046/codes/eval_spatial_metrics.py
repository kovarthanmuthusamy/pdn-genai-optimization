"""Off-anchor eval with spatial metrics (Pearson r, peak location) + early-stop hook."""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from experiments.exp046.codes.spatial_metrics import peak_loc_err, pearson_fg
from src_vae.others.pi_freq_utils import pi_freq_mhz_to_norm


def _fg_mse(recon: torch.Tensor, target: torch.Tensor, bg: float, margin: float = 0.5) -> torch.Tensor:
    fg = (target > bg + margin).float()
    err = (recon - target).pow(2)
    n = fg.sum(dim=(1, 2, 3)).clamp(min=1.0)
    return (err * fg).sum(dim=(1, 2, 3)) / n


@torch.inference_mode()
def run_off_anchor_eval_spatial(
    model: torch.nn.Module,
    val_loader,
    *,
    bg: float,
    off_anchor_mhz: tuple[float, ...] = (100.0, 175.0, 330.0, 400.0),
    max_batches: int = 32,
    device: str | torch.device = "cuda",
    out_csv: Path | str | None = None,
    ckpt_dir: Path | str | None = None,
    early_stop_state: dict | None = None,
    early_stop_mhz: float = 330.0,
    early_stop_patience: int = 3,
    early_stop_min_epoch: int = 150,
) -> list[dict]:
    """Encode path @ off-anchor MHz with spatial metrics per MHz/kind."""
    model.eval()
    base = getattr(model, "_orig_mod", model)
    rows: list[dict] = []
    buckets: dict[tuple[str, float], dict[str, list[float]]] = {}

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
        z_enc_native, _, _, _ = base.encode(hm_enc, occ, imp, K, pi_native)
        skips_native = base._last_heatmap_skips

        for mhz in off_anchor_mhz:
            pi_t = torch.full((hm.shape[0],), pi_freq_mhz_to_norm(mhz), device=device)
            rh_enc, _, _ = base.decode(
                z_enc_native, K, pi_t, occupancy=occ, heatmap_skips=skips_native,
            )
            rh_lay, _, _ = base.decode(z_layout, K, pi_t, occupancy=occ)

            for kind, recon in (("encode_cross", rh_enc), ("layout_cross", rh_lay)):
                mse = _fg_mse(recon, hm, bg)
                pr = pearson_fg(recon, hm, bg)
                pl = peak_loc_err(recon, hm, bg)
                for j in range(mse.shape[0]):
                    key = (kind, float(mhz))
                    b = buckets.setdefault(key, {"mse": [], "pearson": [], "peak_loc": []})
                    b["mse"].append(float(mse[j].item()))
                    b["pearson"].append(float(pr[j].item()))
                    b["peak_loc"].append(float(pl[j].item()))

    for (kind, mhz), vals in sorted(buckets.items()):
        if not vals["mse"]:
            continue
        rows.append({
            "mhz": mhz,
            "kind": kind,
            "n": len(vals["mse"]),
            "hm_fg_mse_mean": sum(vals["mse"]) / len(vals["mse"]),
            "pearson_fg_mean": sum(vals["pearson"]) / len(vals["pearson"]),
            "peak_loc_err_mean": sum(vals["peak_loc"]) / len(vals["peak_loc"]),
        })

    if out_csv is not None:
        path = Path(out_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["mhz", "kind", "n", "hm_fg_mse_mean", "pearson_fg_mean", "peak_loc_err_mean"]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"  Off-anchor spatial eval → {path}")

    for r in rows:
        print(
            f"    {r['kind']:14s} @ {r['mhz']:5.0f} MHz  "
            f"MSE={r['hm_fg_mse_mean']:.4f}  r={r['pearson_fg_mean']:.3f}  "
            f"peakΔ={r['peak_loc_err_mean']:.3f}  n={r['n']}",
        )

    if early_stop_state is not None and out_csv is not None:
        _maybe_early_stop(
            rows,
            early_stop_state,
            epoch=_epoch_from_csv_path(out_csv),
            mhz=early_stop_mhz,
            patience=early_stop_patience,
            min_epoch=early_stop_min_epoch,
            ckpt_dir=ckpt_dir,
            model=model,
        )

    return rows


def _epoch_from_csv_path(out_csv: Path | str) -> int:
    stem = Path(out_csv).stem
    if stem.startswith("off_anchor_eval_epoch_"):
        try:
            return int(stem.rsplit("_", 1)[-1])
        except ValueError:
            pass
    return 0


def _maybe_early_stop(
    rows: list[dict],
    state: dict,
    *,
    epoch: int,
    mhz: float,
    patience: int,
    min_epoch: int = 150,
    ckpt_dir: Path | str | None,
    model: torch.nn.Module,
) -> None:
    if patience <= 0 or epoch < min_epoch:
        return

    enc = next(
        (r for r in rows if r["kind"] == "encode_cross" and abs(r["mhz"] - mhz) < 0.1),
        None,
    )
    if enc is None:
        return

    mse = float(enc["hm_fg_mse_mean"])
    best = float(state.get("best_mse", float("inf")))
    if mse < best:
        state["best_mse"] = mse
        state["best_epoch"] = epoch
        state["best_pearson"] = float(enc["pearson_fg_mean"])
        state["patience"] = 0
        if ckpt_dir is not None:
            path = Path(ckpt_dir) / "best_encode_cross_spatial.pt"
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "metrics": enc}, path)
            print(
                f"  ★ best encode_cross @ {mhz:.0f} MHz ep {epoch}: "
                f"MSE={mse:.4f} r={enc['pearson_fg_mean']:.3f} → {path.name}",
            )
    else:
        state["patience"] = int(state.get("patience", 0)) + 1
        if state["patience"] >= patience:
            c = state.get("config")
            if c is not None:
                c._early_stopped = True
            state["stopped"] = True
            print(
                f"  *** EARLY STOP @ ep {epoch}: encode_cross@{mhz:.0f}MHz "
                f"no improve for {patience} checkpoints "
                f"(best ep {state.get('best_epoch', '?')} MSE={best:.4f})",
            )
