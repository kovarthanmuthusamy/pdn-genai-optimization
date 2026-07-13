"""Train VAE for exp046 — bounded data + spatial losses + U-Net skips + occ tower."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

import sys
from pathlib import Path

_REPO_BOOT = Path(__file__).resolve().parents[3]
if str(_REPO_BOOT) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOT))

from repo_paths import REPO_ROOT as PROJECT_ROOT, setup_path
setup_path()

import experiments.exp038_true_multi.codes.train_vae_simple as _tr
from experiments.exp046.codes.run_epoch_encode import _run_epoch as _run_epoch_encode
from experiments.exp046.codes.spatial_metrics import peak_loc_loss, pearson_fg_loss
from experiments.exp046.codes.synthetic_freq_blend import maybe_apply_synthetic_blend
from experiments.exp046.codes.vae_poe_freq import MultiInputVAEPoeFreq

_orig_prepare = _tr._prepare_batch
_orig_heatmap_loss = _tr.heatmap_loss


@dataclass
class Config(_tr.Config):
    synthetic_blend_prob: float = 0.08
    use_freq_poe_expert: bool = True
    use_heatmap_unet_skips: bool = True
    use_occ_spatial_decoder: bool = True
    use_occ_spatial_tower: bool = True
    occ_spatial_ch: int = 8
    cross_freq_use_encode_z: bool = True
    encode_native_teacher_skips: bool = True
    layout_train_prob: float = 0.15
    heatmap_private_dim: int = 15
    heatmap_pearson_weight: float = 0.5
    heatmap_peak_loc_weight: float = 0.3
    use_layout_private_head: bool = True
    latent_distill_weight: float = 1.0
    latent_distill_private_mult: float = 3.0
    freeze_occ_imp_epoch: int = 300
    early_stop_encode_cross_mhz: float = 330.0
    early_stop_encode_cross_patience: int = 4
    early_stop_min_epoch: int = 150
    _early_stopped: bool = False


_ACTIVE_TRAIN_CONFIG: list[Config] = []
_SPATIAL_EARLY_STOP: dict = {}


def build_vae_model(c: Config) -> MultiInputVAEPoeFreq:
    model = MultiInputVAEPoeFreq(
        latent_dim=c.latent_dim,
        cond_dim=c.cond_dim,
        heatmap_private_dim=c.heatmap_private_dim,
        modality_dropout=c.modality_dropout,
        freq_fourier_features=c.freq_fourier_features,
        use_heatmap_film=c.use_heatmap_film,
        use_freq_poe_expert=bool(getattr(c, "use_freq_poe_expert", True)),
        use_heatmap_unet_skips=bool(getattr(c, "use_heatmap_unet_skips", True)),
        use_occ_spatial_decoder=bool(getattr(c, "use_occ_spatial_decoder", True)),
        use_occ_spatial_tower=bool(getattr(c, "use_occ_spatial_tower", True)),
        occ_spatial_ch=int(getattr(c, "occ_spatial_ch", 8)),
        use_layout_private_head=bool(getattr(c, "use_layout_private_head", True)),
    )
    c._model_ref = model
    print(
        f"  exp046 model: U-Net skips={c.use_heatmap_unet_skips}  "
        f"freq PoE private d={c.heatmap_private_dim}  "
        f"occ_spatial={c.use_occ_spatial_decoder}  occ_tower={c.use_occ_spatial_tower}\n"
        f"    layout_private_head={c.use_layout_private_head}  "
        f"distill_w={c.latent_distill_weight} (private×{c.latent_distill_private_mult})  "
        f"layout_train_prob={c.layout_train_prob}\n"
        f"    freeze_occ_imp_epoch={c.freeze_occ_imp_epoch}",
    )
    return model


def _heatmap_loss_with_spatial(
    recon: torch.Tensor,
    target: torch.Tensor,
    c: _tr.Config,
    ps: float,
    *,
    dynrange_weight: float | None = None,
    lite: bool = False,
) -> torch.Tensor:
    """Standard bounded heatmap loss + optional Pearson + peak-loc."""
    base_loss = _orig_heatmap_loss(recon, target, c, ps, dynrange_weight=dynrange_weight, lite=lite)

    pearson_w = float(getattr(c, "heatmap_pearson_weight", 0.0))
    peak_loc_w = float(getattr(c, "heatmap_peak_loc_weight", 0.0))

    if pearson_w <= 0 and peak_loc_w <= 0:
        return base_loss

    bg = c.background_value + 0.5
    fg = (target > bg).float()

    if pearson_w > 0:
        base_loss = base_loss + pearson_w * pearson_fg_loss(recon, target, fg)

    if peak_loc_w > 0 and not lite:
        base_loss = base_loss + peak_loc_w * peak_loc_loss(recon, target, fg, bg)

    return base_loss


def _prepare_batch(batch: dict, c: _tr.Config) -> tuple:
    return _orig_prepare(maybe_apply_synthetic_blend(batch, c), c)


_FROZEN_OCC_IMP: dict = {"done": False}

_orig_phase_weights = _tr._phase_weights


def _phase_weights_with_freeze(epoch: int, c: Config) -> dict:
    w = _orig_phase_weights(epoch, c)
    freeze_ep = int(getattr(c, "freeze_occ_imp_epoch", 0))
    if freeze_ep > 0 and epoch >= freeze_ep and not _FROZEN_OCC_IMP["done"]:
        if _ACTIVE_TRAIN_CONFIG:
            _freeze_occ_imp_decoders(_ACTIVE_TRAIN_CONFIG[0])
    return w


def _freeze_occ_imp_decoders(c: Config) -> None:
    """Freeze occupancy and impedance decoder parameters to reduce gradient competition."""
    if _FROZEN_OCC_IMP["done"]:
        return
    model = getattr(c, "_model_ref", None)
    if model is None:
        return
    frozen = 0
    for name, param in model.named_parameters():
        if any(tag in name for tag in ("occupancy_dec", "impedance_dec", "occ_fc", "imp_fc")):
            param.requires_grad = False
            frozen += 1
    _FROZEN_OCC_IMP["done"] = True
    print(f"  >> Froze {frozen} occ/imp decoder params at epoch {int(getattr(c, 'freeze_occ_imp_epoch', 0))}")


def _patch_training() -> None:
    _tr.Config = Config
    _tr.build_vae_model = build_vae_model
    _tr._prepare_batch = _prepare_batch
    _tr._run_epoch = _run_epoch_encode
    _tr.heatmap_loss = _heatmap_loss_with_spatial
    _tr._phase_weights = _phase_weights_with_freeze
    _patch_spatial_eval()

    _orig_apply = _tr._apply_yaml_config

    def _apply_yaml_config_track(c: Config) -> None:
        _orig_apply(c)
        _ACTIVE_TRAIN_CONFIG.clear()
        _ACTIVE_TRAIN_CONFIG.append(c)
        _SPATIAL_EARLY_STOP.clear()
        _SPATIAL_EARLY_STOP.update({
            "config": c,
            "best_mse": float("inf"),
            "patience": 0,
            "stopped": False,
        })

    _tr._apply_yaml_config = _apply_yaml_config_track


def _patch_spatial_eval() -> None:
    import experiments.exp038_true_multi.codes.eval_cross_freq as _eval_cf
    from experiments.exp046.codes.eval_spatial_metrics import run_off_anchor_eval_spatial

    if getattr(_eval_cf, "_exp046_spatial_patched", False):
        return

    def _spatial_off_anchor_eval(
        model,
        val_loader,
        *,
        bg: float,
        off_anchor_mhz=(100.0, 175.0, 330.0, 400.0),
        max_batches: int = 30,
        device: str | torch.device = "cuda",
        out_csv=None,
    ):
        c = _ACTIVE_TRAIN_CONFIG[0] if _ACTIVE_TRAIN_CONFIG else None
        ckpt_dir = None
        if c is not None:
            exp = Path(os.environ.get("VAE_EXPERIMENT_DIR", Path(__file__).resolve().parents[1]))
            ckpt_dir = exp / "checkpoints"
        return run_off_anchor_eval_spatial(
            model,
            val_loader,
            bg=bg,
            off_anchor_mhz=off_anchor_mhz,
            max_batches=max_batches,
            device=device,
            out_csv=out_csv,
            ckpt_dir=ckpt_dir,
            early_stop_state=_SPATIAL_EARLY_STOP,
            early_stop_mhz=float(getattr(c, "early_stop_encode_cross_mhz", 330.0)) if c else 330.0,
            early_stop_patience=int(getattr(c, "early_stop_encode_cross_patience", 4)) if c else 4,
            early_stop_min_epoch=int(getattr(c, "early_stop_min_epoch", 150)) if c else 150,
        )

    _eval_cf.run_off_anchor_eval = _spatial_off_anchor_eval
    _eval_cf._exp046_spatial_patched = True


def _print_training_summary(c: _tr.Config) -> None:
    enc_pct = (1.0 - float(c.layout_train_prob)) * 100.0
    print(
        "  exp046 training (bounded + spatial losses):\n"
        f"    dataset: bounded z-score  clip_recon={c.heatmap_clip_recon}\n"
        f"    spatial: pearson_w={c.heatmap_pearson_weight}  peak_loc_w={c.heatmap_peak_loc_weight}  "
        f"grad_w={c.heatmap_grad_weight}  lap_w={c.heatmap_lap_weight}\n"
        f"    encode_native_teacher_skips={getattr(c, 'encode_native_teacher_skips', True)}  "
        f"occ_tower={getattr(c, 'use_occ_spatial_tower', True)}\n"
        f"    layout_train_prob={c.layout_train_prob} (~{enc_pct:.0f}% encode batches)\n"
        f"    cross_freq_use_encode_z={c.cross_freq_use_encode_z}  "
        f"cross_freq_weight={c.cross_freq_weight} from ep {c.cross_freq_start_epoch}\n"
        f"    early_stop: encode_cross@{c.early_stop_encode_cross_mhz}MHz "
        f"patience={c.early_stop_encode_cross_patience} ckpts (min ep {c.early_stop_min_epoch})",
    )


def _restore_early_stop_on_resume(c: Config) -> None:
    """Restore early-stop baseline from best off-anchor eval CSV when resuming."""
    if not c.resume_checkpoint:
        return
    exp = Path(os.environ.get("VAE_EXPERIMENT_DIR", Path(__file__).resolve().parents[1]))
    metrics_dir = exp / "metrics"
    mhz = float(getattr(c, "early_stop_encode_cross_mhz", 330.0))
    min_ep = int(getattr(c, "early_stop_min_epoch", 150))
    resume_ep = int(c.resume_checkpoint) if isinstance(c.resume_checkpoint, int) else 9999
    best_mse = float("inf")
    best_ep = 0
    best_pearson = 0.0

    import csv
    for csv_path in sorted(metrics_dir.glob("off_anchor_eval_epoch_*.csv")):
        try:
            ep = int(csv_path.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if ep < min_ep or ep > resume_ep:
            continue
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("kind") == "encode_cross" and abs(float(row["mhz"]) - mhz) < 0.1:
                    mse = float(row["hm_fg_mse_mean"])
                    if mse < best_mse:
                        best_mse = mse
                        best_ep = ep
                        best_pearson = float(row.get("pearson_fg_mean", 0.0))

    if best_ep <= 0:
        return

    _SPATIAL_EARLY_STOP.update({
        "config": c,
        "best_mse": best_mse,
        "best_epoch": best_ep,
        "best_pearson": best_pearson,
        "patience": 0,
        "stopped": False,
    })
    c._early_stopped = False
    print(
        f"  early-stop state restored: best ep {best_ep} @ {mhz:.0f}MHz "
        f"MSE={best_mse:.4f} r={best_pearson:.3f}",
    )

    # Rebuild best_encode_cross_spatial.pt from the best epoch checkpoint if corrupted.
    best_ckpt = exp / "checkpoints" / f"checkpoint_epoch_{best_ep}.pt"
    best_spatial = exp / "checkpoints" / "best_encode_cross_spatial.pt"
    if best_ckpt.is_file():
        ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        metrics = {
            "mhz": mhz,
            "kind": "encode_cross",
            "hm_fg_mse_mean": best_mse,
            "pearson_fg_mean": best_pearson,
        }
        torch.save({
            "epoch": best_ep,
            "model_state_dict": ckpt["model_state_dict"],
            "metrics": metrics,
        }, best_spatial)
        print(f"  restored {best_spatial.name} from checkpoint_epoch_{best_ep}.pt")


def train_vae() -> None:
    _patch_training()
    c = Config()
    _tr._apply_yaml_config(c)
    _restore_early_stop_on_resume(c)
    _print_training_summary(c)
    _tr.train_vae()


def main() -> None:
    exp_dir = Path(__file__).resolve().parents[1]
    os.environ.setdefault("VAE_EXPERIMENT_DIR", str(exp_dir))
    cfg_path = exp_dir / "config.yaml"
    if cfg_path.is_file():
        lines = [
            ln for ln in cfg_path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        data_dir = json.loads("\n".join(lines)).get("data_dir")
        if data_dir:
            os.environ.setdefault("VAE_DATA_DIR", str(data_dir))
    else:
        os.environ.setdefault(
            "VAE_DATA_DIR",
            str(_PROJECT_ROOT / "data_multi_norm"),
        )
    train_vae()


if __name__ == "__main__":
    main()
