"""Off-anchor eval — spatial metrics, append rows to one CSV (epoch column)."""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from experiments.exp057_structured_graph.codes.spatial_metrics import peak_loc_err, pearson_fg
from src_vae.others.pi_freq_utils import pi_freq_mhz_to_norm

OFF_ANCHOR_CSV_NAME = "off_anchor_eval.csv"
CSV_FIELDS = (
    "epoch",
    "mhz",
    "kind",
    "n",
    "hm_fg_mse_mean",
    "pearson_fg_mean",
    "peak_loc_err_mean",
)


def _fg_mse(recon: torch.Tensor, target: torch.Tensor, bg: float, margin: float = 0.5) -> torch.Tensor:
    fg = (target > bg + margin).float()
    err = (recon - target).pow(2)
    n = fg.sum(dim=(1, 2, 3)).clamp(min=1.0)
    return (err * fg).sum(dim=(1, 2, 3)) / n


def epoch_from_off_anchor_path(out_csv: Path | str | None) -> int:
    stem = Path(out_csv).stem if out_csv is not None else ""
    if stem.startswith("off_anchor_eval_epoch_"):
        try:
            return int(stem.rsplit("_", 1)[-1])
        except ValueError:
            pass
    return 0


def append_off_anchor_rows(path: Path, epoch: int, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})


@torch.inference_mode()
def run_off_anchor_eval_spatial(
    model: torch.nn.Module,
    val_loader,
    *,
    bg: float,
    off_anchor_mhz: tuple[float, ...] = (100.0, 270.0, 400.0),
    max_batches: int = 12,
    device: str | torch.device = "cuda",
    out_csv: Path | str | None = None,
    epoch: int = 0,
    use_encode_skips: bool = False,
    use_binary_occupancy: bool = True,
) -> list[dict]:
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

        from experiments.exp057_structured_graph.codes.occupancy_binary import occupancy_for_heatmap_decode

        occ_dec = occupancy_for_heatmap_decode(occ, K, force_binary=use_binary_occupancy) if use_binary_occupancy else occ
        z_layout = base.encode_layout_latent(occ_dec, imp, K, pi_native)
        z_enc_native, _, _, _ = base.encode(hm_enc, occ_dec, imp, K, pi_native)
        skips_native = base._last_heatmap_skips

        for mhz in off_anchor_mhz:
            pi_t = torch.full((hm.shape[0],), pi_freq_mhz_to_norm(mhz), device=device)
            rh_enc, _, _ = base.decode(
                z_enc_native, K, pi_t, occupancy=occ_dec, heatmap_skips=skips_native if use_encode_skips else None,
            )
            rh_lay, _, _ = base.decode(z_layout, K, pi_t, occupancy=occ_dec)

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
            "epoch": epoch,
            "mhz": mhz,
            "kind": kind,
            "n": len(vals["mse"]),
            "hm_fg_mse_mean": sum(vals["mse"]) / len(vals["mse"]),
            "pearson_fg_mean": sum(vals["pearson"]) / len(vals["pearson"]),
            "peak_loc_err_mean": sum(vals["peak_loc"]) / len(vals["peak_loc"]),
        })

    if out_csv is not None and rows:
        path = Path(out_csv)
        if path.name.startswith("off_anchor_eval_epoch_"):
            path = path.parent / OFF_ANCHOR_CSV_NAME
        append_off_anchor_rows(path, epoch, rows)
        enc = [r for r in rows if r["kind"] == "encode_cross"]
        parts = [
            f"{int(r['mhz'])}MHz:r={r['pearson_fg_mean']:.2f}"
            for r in sorted(enc, key=lambda x: x["mhz"])
        ]
        print(
            f"  Off-anchor ep {epoch} → {path.name}  encode_cross: " + "  ".join(parts),
            flush=True,
        )

    return rows
