"""Train VAE for exp058 Graph VAE — tier_a + peak/valley extrema losses."""

from __future__ import annotations
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from repo_paths import REPO_ROOT as PROJECT_ROOT, setup_path

setup_path()

import torch

import experiments.exp058_asymmetric_kl.codes.train_core as _tr
from experiments.exp058_asymmetric_kl.codes.exp058_common import EXP_DIR, load_yaml_config, vae_model_kwargs
from experiments.exp058_asymmetric_kl.codes.run_epoch_encode import _run_epoch as _run_epoch_encode
from experiments.exp058_asymmetric_kl.codes.heatmap_peak_losses import ExtremaSide, heatmap_extrema_log1p_both, heatmap_loss_pearson_grad
from experiments.exp058_asymmetric_kl.codes.spatial_metrics import extrema_loc_sharp_loss
from experiments.exp058_asymmetric_kl.codes.vae_poe_freq import MultiInputVAEPoeFreq
from experiments.exp058_asymmetric_kl.codes.distributed_train import is_main_process

_EXTREMA_LOC = (
    ("heatmap_peak_loc_weight", "heatmap_peak_loc_temperature", "peak"),
    ("heatmap_valley_loc_weight", "heatmap_valley_loc_temperature", "valley"),
)
_LOG1P_SIDES: tuple[tuple[ExtremaSide, str, str, str], ...] = (
    ("peak", "heatmap_peak_log1p_weight", "heatmap_focus_peak_log1p_mult", "heatmap_peak_log1p_loss"),
    ("valley", "heatmap_valley_log1p_weight", "heatmap_focus_valley_log1p_mult", "heatmap_valley_log1p_loss"),
)
_SCHEDULE_KEYS = (
    "cross_freq_start_epoch", "impedance_peak_start_epoch", "impedance_peak_focus_epoch",
    "beta_end_epoch", "heatmap_focus_start_epoch", "modality_dropout_protect_heatmap_epoch",
)
_SCHEDULE_EPS: set[int] = set()


@dataclass
class Config(_tr.Config):
    # exp055 additions (not present in exp038 base Config)
    cross_freq_layout_mix_prob: float = 0.45
    eval_off_anchor_interval: int = 50
    # Distillation (layout student → encode teacher)
    latent_distill_weight: float = 2.0
    latent_distill_private_mult: float = 3.0
    latent_distill_logvar_weight: float = 0.3
    output_distill_weight: float = 2.5
    # exp055: training loop / stability knobs
    cross_freq_use_encode_z: bool = True
    encode_native_teacher_skips: bool = True
    cross_freq_use_encode_skips: bool = False  # off: decode alt MHz without native-MHz U-Net skips
    occupancy_binary_decode: bool = True  # top-K binary occ before heatmap decode (CAD-aligned)
    occupancy_binary_ste_train: bool = False  # straight-through top-K during training decode
    training_force_fp32: bool = True
    training_sanitize_grads: bool = False
    training_skip_nonfinite_batches: bool = True
    training_grad_clip_norm: float = 1.0
    training_logvar_clamp_min: float = -10.0
    training_logvar_clamp_max: float = 10.0
    training_recon_z_soft_floor: float = -8.0
    training_recon_z_soft_cap: float | None = None
    training_recon_z_cap_mult: float = 1.05
    training_loss_finite_cap: float = 1e6
    impedance_separate_lr: bool = True
    use_ddp: bool = True
    ddp_base_batch_size: int = 160
    ddp_linear_lr_scale: bool = True
    # heatmap tier_a weights (Pearson+grad)
    heatmap_pearson_weight: float = 0.75
    heatmap_grad_vector_weight: float = 2.0
    heatmap_grad_direction_weight: float = 1.0
    heatmap_grad_direction_min_mag: float = 0.08
    heatmap_grad_huber_delta: float = 0.5
    heatmap_peak_centroid_top_q: float = 0.95
    heatmap_valley_centroid_bottom_q: float = 0.05
    # top-region Huber (replaces centroid): amplitude + position on GT extreme-percentile pixels
    heatmap_peak_topregion_weight: float = 0.0
    heatmap_valley_topregion_weight: float = 0.0
    heatmap_peak_topregion_q: float = 0.95
    heatmap_valley_topregion_q: float = 0.05
    heatmap_topregion_huber_delta: float = 0.5


def _on_train_epoch_start(epoch_1based: int, c: Config, train_ld) -> None:
    del train_ld, c
    if epoch_1based in _SCHEDULE_EPS:
        print(f"  >> milestone epoch {epoch_1based}", flush=True)


def build_vae_model(c: Config) -> MultiInputVAEPoeFreq:
    model = MultiInputVAEPoeFreq(**vae_model_kwargs(c))
    model.binary_occupancy_decode = bool(getattr(c, "occupancy_binary_decode", True))
    model.occupancy_binary_ste = bool(getattr(c, "occupancy_binary_ste_train", False))
    _SCHEDULE_EPS.clear()
    _SCHEDULE_EPS.update(int(getattr(c, k, 0)) for k in _SCHEDULE_KEYS if int(getattr(c, k, 0)) > 0)
    sched = ", ".join(f"ep{e}" for e in sorted(_SCHEDULE_EPS))
    n = sum(p.numel() for p in model.parameters())
    if is_main_process():
        print(f"Model: {n/1e6:.1f}M params  latent={c.latent_dim}  layout_p={c.layout_train_prob}  "
              f"cf_mix={c.cross_freq_layout_mix_prob}  occ_bin={model.binary_occupancy_decode}  "
              f"milestones=[{sched}]", flush=True)
    return model


def _extrema_loc_losses(recon, target, c):
    bg = c.background_value + 0.5
    fg = (target > bg).float()
    out = recon.new_zeros(recon.shape[0])
    for w_attr, t_attr, side in _EXTREMA_LOC:
        w = float(getattr(c, w_attr, 0))
        if w > 0:
            out = out + w * extrema_loc_sharp_loss(recon, target, fg, bg, side=side, temperature=float(getattr(c, t_attr, 0.03)))
    return out


def _heatmap_loss_with_spatial(recon, target, c, ps, *, lite=False, **_):
    del ps
    loss = heatmap_loss_pearson_grad(recon, target, c, lite=lite)
    return loss if lite else loss + _extrema_loc_losses(recon, target, c)


def _on_stats_loaded(c: Config, raw: dict) -> None:
    from src_vae.others.norm_stats import NormStatsBundle
    bundle = NormStatsBundle.load(c.data_dir)
    c._norm_stats, c._hm_log_mean, c._hm_log_std = bundle, float(bundle.heatmap.log_mean), float(bundle.heatmap.log_std)
    if not bundle.heatmap.is_unbounded():
        raise ValueError(f"exp055 requires unbounded heatmap stats; got {bundle.heatmap.norm_mode!r}")
    c.amp = "off"  # permanent fp32 training (AMP train path caused NaN grads)


def _log1p_weight(c, weight_attr, focus_attr, epoch, focus_ep):
    w = float(getattr(c, weight_attr, 0))
    return w * float(getattr(c, focus_attr, 1)) if focus_ep > 0 and epoch >= focus_ep else w


def _apply_extrema_log1p_losses(losses, recon_hm, tgt_hm, c, *, epoch, pi_freq, focus_ep):
    active = [(s, wa, fa, lk) for s, wa, fa, lk in _LOG1P_SIDES if _log1p_weight(c, wa, fa, epoch, focus_ep) > 0]
    if not active:
        return
    both = heatmap_extrema_log1p_both(recon_hm, tgt_hm, c, pi_freq=pi_freq, sides=tuple(s[0] for s in active))
    for side, wa, fa, lk in active:
        if side not in both:
            continue
        per, _ = both[side]
        w = _log1p_weight(c, wa, fa, epoch, focus_ep)
        ext = per.reshape(-1).mean()
        losses[lk] = ext
        losses["total_loss"] = losses["total_loss"] + w * ext
        losses["heatmap_loss"] = losses["heatmap_loss"] + w * ext


_orig_vae_loss = _tr.vae_loss


def _vae_loss_exp055(recon_hm, recon_occ, recon_imp, tgt_hm, tgt_occ, tgt_imp, mu, logvar, beta, c,
                     expert_stats, *, epoch, K, physics, pw, pi_freq, imp_log_std, ps, apply_k, weight_overrides=None):
    from experiments.exp058_asymmetric_kl.codes.training_guard import clamp_logvar, sanitize_loss_dict
    hm_ps = _tr.heatmap_loss(recon_hm, tgt_hm, c, ps)
    logvar = clamp_logvar(logvar, c)
    prev = _tr.heatmap_loss
    _tr.heatmap_loss = lambda *a, **kw: hm_ps
    try:
        losses = _orig_vae_loss(recon_hm, recon_occ, recon_imp, tgt_hm, tgt_occ, tgt_imp, mu, logvar, beta, c,
            expert_stats, epoch=epoch, K=K, physics=None, pw=None, pi_freq=pi_freq, imp_log_std=imp_log_std,
            ps=ps, apply_k=False, weight_overrides=weight_overrides)
    finally:
        _tr.heatmap_loss = prev
    losses["heatmap_loss_tier_a"] = losses["heatmap_loss"].detach()
    _apply_extrema_log1p_losses(losses, recon_hm, tgt_hm, c, epoch=epoch, pi_freq=pi_freq,
                                focus_ep=int(getattr(c, "heatmap_focus_start_epoch", 0)))
    return sanitize_loss_dict(losses, c)


def _append_heatmap_peak_split_csv(metrics_dir, epoch, tr, val):
    path = metrics_dir / "heatmap_peak_split.csv"
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["epoch", "train_heatmap_loss", "train_tier_a", "train_peak_log1p", "train_valley_log1p",
                        "val_heatmap_loss", "val_tier_a", "val_peak_log1p", "val_valley_log1p"])
        w.writerow([epoch, tr.get("heatmap_loss", ""), tr.get("heatmap_loss_tier_a", ""),
                    tr.get("heatmap_peak_log1p_loss", ""), tr.get("heatmap_valley_log1p_loss", ""),
                    val.get("heatmap_loss", "") if val else "", val.get("heatmap_loss_tier_a", "") if val else "",
                    val.get("heatmap_peak_log1p_loss", "") if val else "", val.get("heatmap_valley_log1p_loss", "") if val else ""])


def _apply_yaml_exp055(c: Config) -> None:
    """Load config.yaml then optional ``VAE_CONFIG_PATH`` (AL fine-tune runtime)."""
    config_paths: list[Path] = [Path(c.experiment_dir) / "config.yaml"]
    override = os.environ.get("VAE_CONFIG_PATH")
    if override:
        op = Path(override)
        if op.is_file() and op not in config_paths:
            config_paths.append(op)
    if not config_paths[0].is_file():
        raise FileNotFoundError(f"exp058 requires config.yaml at {config_paths[0]}")
    explicit: set[str] = set()
    for cfg_path in config_paths:
        if not cfg_path.is_file():
            continue
        data = _tr.load_experiment_config(cfg_path)
        explicit |= {
            k for k, v in data.items()
            if k.endswith("_epoch") and isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        for key, val in data.items():
            if key.startswith("_"):
                continue
            if key == "eval_off_anchor_mhz" and isinstance(val, list):
                setattr(c, key, tuple(float(x) for x in val))
            else:
                setattr(c, key, val)
        print(f"Loaded config from {cfg_path.name} ({len(data)} keys)")
    _tr.apply_curriculum_epochs(c, skip_epoch_keys=explicit)
    _tr.print_curriculum_summary(c)
    c.on_train_epoch_start = _on_train_epoch_start
    from experiments.exp058_asymmetric_kl.codes.eval_off_anchor import set_off_anchor_config
    set_off_anchor_config(c)





def _impedance_weights_exp055(c: Config):
    from experiments.exp058_asymmetric_kl.codes.impedance_spectrum_loss import ImpedanceSpectrumWeights
    return ImpedanceSpectrumWeights(
        deriv_weight=c.impedance_deriv_weight,
        topk_k=c.impedance_topk_k,
        topk_weight=c.impedance_topk_weight,
        under_penalty=c.impedance_under_penalty,
        freq_weight_alpha=c.impedance_freq_weight_alpha,
        peak_index_weight=c.impedance_peak_index_weight,
        num_peaks=c.impedance_num_peaks,
    )


def _impedance_loss_exp055(recon, target, c, imp_log_std, ps, *, epoch):
    from experiments.exp058_asymmetric_kl.codes.impedance_spectrum_loss import impedance_spectrum_loss
    return impedance_spectrum_loss(
        recon, target,
        imp_log_std=imp_log_std,
        penalty_scale=ps,
        peak_scale=_tr._peak_loss_scale(epoch, c),
        w=_impedance_weights_exp055(c),
    )



def _patch_training() -> None:
    from experiments.exp058_asymmetric_kl.codes import dataloader_multifreq as dm
    _tr.Config, _tr.build_vae_model, _tr._run_epoch = Config, build_vae_model, _run_epoch_encode
    _tr.heatmap_loss, _tr.vae_loss, _tr._on_stats_loaded = _heatmap_loss_with_spatial, _vae_loss_exp055, _on_stats_loaded
    _tr.LOSS_KEYS = _tr.LOSS_KEYS + ("heatmap_peak_log1p_loss", "heatmap_valley_log1p_loss", "heatmap_loss_tier_a")
    _tr.create_multifreq_data_loaders = dm.create_multifreq_data_loaders
    _tr._apply_yaml_config = _apply_yaml_exp055
    _tr.impedance_loss = _impedance_loss_exp055

    _orig_opt, _orig_split = _tr._build_optimizer, _tr._append_impedance_split_csv

    def _build_optimizer_careful_imp(model, physics, c, epoch_start, *, use_grad_scaler=False):
        base, mult, focus_ep = float(c.learning_rate), float(c.impedance_decoder_lr_mult), int(c.impedance_peak_focus_epoch)
        if c.impedance_separate_lr and epoch_start >= focus_ep:
            imp_p = _tr._impedance_module_params(model)
            imp_ids = {id(p) for p in imp_p}
            other = [p for _, p in model.named_parameters() if p.requires_grad and id(p) not in imp_ids]
            imp_lr = base * mult
            print(f"  Careful impedance LR: backbone={base:.2e}  impedance={imp_lr:.2e} (mult={mult})", flush=True)
            return _tr._adamw([{"params": other, "lr": base}, {"params": imp_p, "lr": imp_lr}], lr=base, c=c, use_grad_scaler=use_grad_scaler)
        return _orig_opt(model, physics, c, epoch_start, use_grad_scaler=use_grad_scaler)

    def _append_splits(metrics_dir, epoch, tr, val):
        _orig_split(metrics_dir, epoch, tr, val)
        _append_heatmap_peak_split_csv(metrics_dir, epoch, tr, val)

    _tr._build_optimizer, _tr._append_impedance_split_csv = _build_optimizer_careful_imp, _append_splits



def _patch_quiet_logging() -> None:
    _tr.print_curriculum_summary = lambda _c: None
    _orig_load, _orig_train = _tr.load_checkpoint, _tr.train_vae
    _SKIP = ("epochs ", "params:", "  DataLoader:", "checkpoint_interval=", "  Val: epoch", "  >> Heatmap focus",
             "  imp legacy=", "  phys ri=", "  Note: no checkpoint file")

    def _load_quiet(path, model, optimizer=None, device="cuda", physics=None):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        ep, val_loss = int(ckpt.get("epoch", 0)), float(ckpt.get("val_loss", float("nan")))
        nep = int((ckpt.get("config") or {}).get("num_epochs", 0)) or None
        override = os.environ.get("VAE_CONFIG_PATH")
        if override:
            op = Path(override)
            if op.is_file():
                try:
                    od = _tr.load_experiment_config(op)
                    if od.get("num_epochs"):
                        nep = int(od["num_epochs"])
                except Exception:
                    pass
        msd = model.state_dict()
        compat = {k: v for k, v in ckpt["model_state_dict"].items() if k in msd and v.shape == msd[k].shape}
        if not compat and os.getenv("VAE_ALLOW_PARTIAL_CHECKPOINT") not in {"1", "true", "True"}:
            raise RuntimeError("Checkpoint incompatible — set VAE_ALLOW_PARTIAL_CHECKPOINT=1 to migrate")
        msd.update(compat); model.load_state_dict(msd)
        if optimizer and "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except ValueError:
                print("  (optimizer state skipped — param groups changed)", flush=True)
        print(f"\n{'='*56}\n  RESUMING from checkpoint {Path(path).name}\n  completed epoch: {ep}  →  "
              f"training epochs {ep+1}–{nep or '?'}\n  val_loss: {val_loss:.4f}\n{'='*56}\n", flush=True)
        return ep, val_loss, ckpt.get("config") or {}

    _tr.load_checkpoint = _load_quiet

    try:
        import src_vae.others.heatmap_z_clip as hz
        hz.describe_clip_bounds = lambda _d: ""
    except Exception:
        pass

    def _clean_prev(exp):
        import shutil
        removed = []
        for name in ("checkpoints", "logs", "metrics"):
            p = exp / name
            if p.exists():
                shutil.rmtree(p); removed.append(name)
        if removed:
            print(f"Fresh run — cleared {', '.join(removed)}", flush=True)

    _tr._clean_previous_run = _clean_prev

    def _train_quiet():
        import builtins
        real = builtins.print
        def filt(*args, **kw):
            msg = " ".join(str(a) for a in args)
            if not msg.strip() or msg.startswith("Loaded config overrides from") or any(msg.startswith(p) for p in _SKIP):
                return
            return real(*args, **kw)
        builtins.print = filt
        try:
            _orig_train()
        finally:
            builtins.print = real

    _tr.train_vae = _train_quiet


def train_vae() -> None:
    _patch_training()
    _patch_quiet_logging()
    c = Config()
    _tr._apply_yaml_config(c)
    rc = getattr(c, "resume_checkpoint", None)
    if is_main_process():
        print(f"exp058  epochs={c.num_epochs}  batch={c.batch_size}  amp={c.amp}  ckpt_every={c.checkpoint_interval}  "
          f"resume={'fresh' if not rc else rc}", flush=True)
    _tr.train_vae()


def main() -> None:
    os.environ.setdefault("VAE_EXPERIMENT_DIR", str(EXP_DIR))
    data_dir = load_yaml_config().get("data_dir")
    os.environ.setdefault("VAE_DATA_DIR", str(data_dir or PROJECT_ROOT / "datasets/data_multifreq_train_norm_unbounded"))
    train_vae()


if __name__ == "__main__":
    main()
