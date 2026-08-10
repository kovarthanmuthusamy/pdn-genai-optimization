"""Pydantic training config + shared paths/kwargs for exp059_capacity_freq."""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field
from pydantic.fields import PydanticUndefined


# ── Shared paths / VAE kwargs (formerly exp059_common.py) ─────────────────────

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXP_DIR = Path(__file__).resolve().parents[1]


def load_yaml_config(path: Path | None = None) -> dict[str, Any]:
    path = path or (EXP_DIR / "config.yaml")
    if not path.is_file():
        return {}
    lines = [
        ln for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    return json.loads("\n".join(lines))


def _cfg_get(cfg: dict[str, Any] | Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def vae_model_kwargs(cfg: dict[str, Any] | Any) -> dict[str, Any]:
    g = lambda k, d=None: _cfg_get(cfg, k, d)
    return {
        "latent_dim": int(g("latent_dim", 42)),
        "cond_dim": int(g("cond_dim", 8)),
        "heatmap_private_dim": int(g("heatmap_private_dim", 8)),
        "modality_dropout": float(g("modality_dropout", 0.0)),
        "freq_fourier_features": int(g("freq_fourier_features", 8)),
        "use_heatmap_film": bool(g("use_heatmap_film", True)),
        "use_multiscale_film": bool(g("use_multiscale_film", True)),
        "use_freq_poe_expert": bool(g("use_freq_poe_expert", True)),
        "use_occ_spatial_decoder": bool(g("use_occ_spatial_decoder", True)),
        "use_occ_spatial_tower": bool(g("use_occ_spatial_tower", True)),
        "occ_spatial_ch": int(g("occ_spatial_ch", 8)),
        "use_layout_private_head": bool(g("use_layout_private_head", True)),
        "layout_private_hidden": int(g("layout_private_hidden", 384)),
        "use_layout_private_freq_film": bool(g("use_layout_private_freq_film", True)),
        "graph_hidden_dim": int(g("graph_hidden_dim", 128)),
        "graph_num_layers": int(g("graph_num_layers", 3)),
        "graph_dropout": float(g("graph_dropout", 0.1)),
        "imp_peak_dim": int(g("imp_peak_dim", 8)),
    }


# ── Curriculum / schedule helpers ───────────────────────────────────────────

_FRAC_FIELDS = (
    ("beta_end_epoch", "beta_end_frac"),
    ("beta_phase2_end_epoch", "beta_phase2_end_frac"),
    ("modality_dropout_anneal_epochs", "modality_dropout_anneal_frac"),
    ("physics_critic_warmup_epochs", "physics_critic_warmup_frac"),
    ("physics_slope_anneal_epochs", "physics_slope_anneal_frac"),
    ("penalty_warmup_epochs", "penalty_warmup_frac"),
    ("focal_gamma_warmup_epochs", "focal_gamma_warmup_frac"),
)

_CURRICULUM_FRAC_FIELDS = (
    ("cross_freq_start_epoch", "cross_freq_start_frac"),
    ("modality_dropout_protect_heatmap_epoch", "modality_dropout_protect_heatmap_frac"),
    ("heatmap_focus_start_epoch", "heatmap_focus_start_frac"),
    ("impedance_peak_start_epoch", "impedance_peak_start_frac"),
    ("impedance_peak_ramp_epochs", "impedance_peak_ramp_frac"),
    ("impedance_peak_focus_epoch", "impedance_peak_focus_frac"),
)

_ALL_SCHEDULE_FRAC_PAIRS = _FRAC_FIELDS + _CURRICULUM_FRAC_FIELDS


def apply_curriculum_epochs(c: "TrainConfig", *, skip_epoch_keys: set[str] | None = None) -> None:
    """Map ``*_frac`` → absolute ``*_epoch`` = round(num_epochs * frac)."""
    skip = skip_epoch_keys or set()
    n = int(c.num_epochs)
    for attr_epoch, attr_frac in _ALL_SCHEDULE_FRAC_PAIRS:
        if attr_epoch in skip:
            continue
        frac = getattr(c, attr_frac)
        setattr(c, attr_epoch, round(n * frac))


def restore_curriculum_epochs_from_checkpoint(c: "TrainConfig", ckpt_cfg: dict) -> bool:
    """Keep phase boundaries from the run that wrote the checkpoint (resume-safe)."""
    restored: list[str] = []
    for attr_epoch, _ in _ALL_SCHEDULE_FRAC_PAIRS:
        if attr_epoch in ckpt_cfg and ckpt_cfg[attr_epoch] is not None:
            setattr(c, attr_epoch, int(ckpt_cfg[attr_epoch]))
            restored.append(f"{attr_epoch}={int(ckpt_cfg[attr_epoch])}")
    if restored:
        print("  Curriculum restored from checkpoint (not re-scaled to new num_epochs):")
        print("    " + ", ".join(restored))
        return True
    return False


def clamp_curriculum_for_resume_epoch(c: "TrainConfig", resume_epoch: int) -> None:
    """If resume is already past a phase start, do not push that start into the future."""
    if resume_epoch <= 0:
        return
    for attr_epoch, _ in _ALL_SCHEDULE_FRAC_PAIRS:
        val = int(getattr(c, attr_epoch))
        if val > resume_epoch:
            setattr(c, attr_epoch, resume_epoch)


def print_curriculum_summary(c: "TrainConfig") -> None:
    print(
        f"  Curriculum (of {c.num_epochs} epochs): "
        f"cross_freq@{c.cross_freq_start_epoch}, "
        f"protect_hm@{c.modality_dropout_protect_heatmap_epoch}, "
        f"heatmap_focus@{c.heatmap_focus_start_epoch}, "
        f"imp_peak@{c.impedance_peak_start_epoch}→{c.impedance_peak_focus_epoch} "
        f"(ramp {c.impedance_peak_ramp_epochs}), "
        f"layout_train_prob={c.layout_train_prob}, "
        f"occ_only_encode_prob={c.occ_only_encode_prob}, "
        f"eval_use_occ_only_layout={c.eval_use_occ_only_layout}, "
        f"cross_freq_layout_z_only={c.cross_freq_layout_z_only}",
    )


class TrainConfig(BaseModel):
    """Canonical exp059 train config (YAML + curriculum + runtime hooks)."""

    model_config = ConfigDict(
        extra="allow",  # runtime hooks (_norm_stats, on_train_epoch_start, ...)
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    latent_dim: int = 42
    heatmap_private_dim: int = 8
    cond_dim: int = 8
    num_epochs: int = 850
    batch_size: int = 96
    val_batch_size: int = 0
    learning_rate: float = 2e-5
    lr_min: float = 3e-6
    lr_patience: int = 25
    lr_factor: float = 0.5
    train_split: float = 0.9
    num_workers: int = 8
    reset_lr_on_resume: bool = True
    balance_k: bool = True
    k_balance_power: float = 0.5
    k_balance_smoothing: float = 1e-3
    stratify_by_k: bool = True
    split_by_design: bool = True
    balance_freq: bool = True
    freq_balance_power: float = 1.0
    train_samples_per_epoch: int = 50_000
    al_overlay_data_dir: str | None = None
    al_overlay_sample_weight: float = 25.0
    val_on_checkpoint_only: bool = True
    epoch_print_interval: int = 2
    epoch_log_interval: int = 2
    heatmap_weight: float = 2.75
    cross_freq_weight: float = 1.0
    cross_freq_start_frac: float = 0.05
    cross_freq_start_epoch: int = 0
    modality_dropout_protect_heatmap_frac: float = 0.5
    modality_dropout_protect_heatmap_epoch: int = 0
    freq_fourier_features: int = 8
    use_heatmap_film: bool = True
    freq_jitter_prob: float = 0.3
    freq_jitter_log10_sigma: float = 0.08
    heatmap_phys_p99_weight: float = 0.5
    heatmap_phys_p99_percentile: float = 99.0
    heatmap_focus_start_frac: float = 0.5
    heatmap_focus_start_epoch: int = 0
    heatmap_focus_heatmap_weight: float = 3.5
    heatmap_focus_impedance_weight: float = 1.5
    heatmap_focus_cross_freq_weight: float = 1.0
    heatmap_focus_dynrange_weight: float = 3.0
    heatmap_focus_modality_dropout: float = 0.05
    layout_train_prob: float = 0.4
    occ_only_encode_prob: float = 0.0
    eval_use_occ_only_layout: bool = False
    cross_freq_layout_z_only: bool = False
    eval_off_anchor_mhz: tuple[float, ...] = (80.0, 250.0)
    eval_off_anchor_mhz_weights: dict | None = None
    eval_off_anchor_max_batches: int = 30
    al_finetune_early_stop: bool = False
    al_finetune_early_stop_patience: int = 2
    al_finetune_early_stop_min_delta: float = 0.01
    al_finetune_early_stop_kind: str = "layout_cross"
    occupancy_weight: float = 8.0
    impedance_weight: float = 3.0
    impedance_deriv_weight: float = 1.5
    impedance_topk_k: int = 20
    impedance_topk_weight: float = 7.0
    impedance_under_penalty: float = 2.8
    impedance_concavity_weight: float = 2.5
    impedance_freq_weight_alpha: float = 2.0
    impedance_dual_topk_weight: float = 0.75
    impedance_peak_index_weight: float = 2.5
    impedance_peak_mag_weight: float = 2.0
    impedance_num_peaks: int = 8
    impedance_peak_start_frac: float = 0.5
    impedance_peak_ramp_frac: float = 0.11
    impedance_peak_focus_frac: float = 0.5
    impedance_peak_start_epoch: int = 0
    impedance_peak_ramp_epochs: int = 0
    impedance_peak_focus_epoch: int = 0
    impedance_decoder_lr_mult: float = 4.0
    heatmap_peak_weight: float = 3.0
    heatmap_grad_weight: float = 1.5
    heatmap_lap_weight: float = 2.0
    heatmap_contrast_weight: float = 1.5
    heatmap_contrast_margin: float = 0.5
    heatmap_bg_weight: float = 0.5
    heatmap_dynrange_weight: float = 2.0
    occupancy_focal_gamma: float = 1.5
    focal_gamma_warmup_frac: float = 0.0
    focal_gamma_warmup_epochs: int = 0
    occ_k_consistency_weight: float = 2.0
    use_k_weighting: bool = True
    occ_mid_k_center: float = 25.0
    occ_mid_k_sigma: float = 8.0
    occ_mid_k_boost: float = 0.75
    hm_low_k_threshold: int = 3
    hm_low_k_multiplier: float = 2.0
    free_bits: float = 0.05
    mu_hinge_threshold: float = 4.0
    mu_hinge_weight: float = 0.10
    mu_bias_weight: float = 0.05
    per_expert_kl_weight: float = 0.02
    sigma_reg_weight: float = 1.5
    sigma_reg_target: float = 0.45
    use_beta_annealing: bool = True
    beta_start_epoch: int = 0
    beta_end_frac: float = 0.40
    beta_initial: float = 0.0
    beta_final: float = 0.1
    beta_phase2_final: float = 0.15
    beta_phase2_end_frac: float = 0.80
    beta_end_epoch: int = 0
    beta_phase2_end_epoch: int = 0
    modality_dropout: float = 0.12
    modality_dropout_start: float = 0.08
    modality_dropout_anneal_frac: float = 0.08
    modality_dropout_anneal_epochs: int = 0
    cross_modal_weight: float = 0.85
    cross_modal_update_freq: int = 4
    physics_ri_weight: float = 1.0
    physics_critic_sup_weight: float = 2.0
    physics_ar_weight: float = 0.5
    physics_fg_clip_min: float = -1.04
    physics_critic_warmup_frac: float = 0.05
    physics_slope_anneal_frac: float = 0.10
    physics_critic_warmup_epochs: int = 0
    physics_slope_anneal_epochs: int = 0
    penalty_warmup_frac: float = 0.05
    penalty_warmup_epochs: int = 0
    data_dir: str = Field(default_factory=lambda: os.environ.get("VAE_DATA_DIR", "datasets/data_multifreq_norm"))
    experiment_dir: str = Field(default_factory=lambda: os.environ.get("VAE_EXPERIMENT_DIR", "experiments/exp059_capacity_freq"))
    checkpoint_interval: int = 25
    keep_last_n_checkpoints: int = 20
    resume_checkpoint: int | str | None = None
    recalculate_curriculum_on_resume: bool = False
    background_value: float = -3.6228
    heatmap_z_clip_min: float | None = None
    heatmap_z_clip_max: float | None = None
    heatmap_clip_recon: bool = True
    amp: str = "bf16"
    tf32: bool = True
    compile: bool = True
    compile_mode: str = "reduce-overhead"
    cache_in_ram: bool = True
    use_ddp: bool = True
    ddp_base_batch_size: int = 160
    ddp_linear_lr_scale: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    empty_cache_interval: int = 25
    kan_spline_l1_weight: float = 1e-4
    device: str = Field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    cross_freq_layout_mix_prob: float = 0.45
    eval_off_anchor_interval: int = 50
    latent_distill_weight: float = 2.0
    latent_distill_private_mult: float = 3.0
    latent_distill_logvar_weight: float = 0.3
    output_distill_weight: float = 2.5
    cross_freq_use_encode_z: bool = True
    occupancy_binary_decode: bool = True
    occupancy_binary_ste_train: bool = False
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
    heatmap_pearson_weight: float = 0.75
    heatmap_grad_vector_weight: float = 2.0
    heatmap_grad_direction_weight: float = 1.0
    heatmap_grad_direction_min_mag: float = 0.08
    heatmap_grad_huber_delta: float = 0.5
    heatmap_peak_centroid_top_q: float = 0.95
    heatmap_valley_centroid_bottom_q: float = 0.05
    heatmap_peak_topregion_weight: float = 0.0
    heatmap_valley_topregion_weight: float = 0.0
    heatmap_peak_topregion_q: float = 0.95
    heatmap_valley_topregion_q: float = 0.05
    heatmap_topregion_huber_delta: float = 0.5
    heatmap_peak_loc_weight: float = 0.0
    heatmap_peak_loc_temperature: float = 0.03
    heatmap_valley_loc_weight: float = 0.0
    heatmap_valley_loc_temperature: float = 0.03

    @property
    def log_interval(self) -> int:
        """Alias for checkpoint_interval (external tools / config.yaml)."""
        return self.checkpoint_interval

    def is_cuda(self) -> bool:
        return "cuda" in self.device

    def to_checkpoint_dict(self) -> dict[str, Any]:
        """JSON-serializable snapshot of declared fields only (no runtime extras)."""
        return self.model_dump(mode="json", exclude=set(self.__pydantic_extra__ or {}))


def load_experiment_config_dict(path: Path) -> dict[str, Any]:
    """Parse config.yaml: JSON object with optional full-line ``#`` comments."""
    text = path.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    return json.loads("\n".join(lines)) if lines else {}


def _default_value(field) -> Any:
    if field.default_factory is not None and field.default_factory is not PydanticUndefined:
        return field.default_factory()
    if field.default is not PydanticUndefined:
        return field.default
    return None


def train_config_from_yaml(
    path: Path | None = None,
    *,
    base: TrainConfig | None = None,
    apply_curriculum: bool = True,
) -> TrainConfig:
    """Build/merge a TrainConfig from YAML (and optional existing instance)."""
    data: dict[str, Any] = {}
    if path is not None and Path(path).is_file():
        data = load_experiment_config_dict(Path(path))
    data = {k: v for k, v in data.items() if not str(k).startswith("_")}
    if "eval_off_anchor_mhz" in data and isinstance(data["eval_off_anchor_mhz"], list):
        data["eval_off_anchor_mhz"] = tuple(float(x) for x in data["eval_off_anchor_mhz"])

    if base is None:
        # Construct defaults without after-validator; apply curriculum once below.
        out = TrainConfig.model_construct(**{
            name: _default_value(field)
            for name, field in TrainConfig.model_fields.items()
        })
    else:
        out = base.model_copy(deep=True)

    explicit = {
        k for k, v in data.items()
        if k.endswith("_epoch") and isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    for key, val in data.items():
        setattr(out, key, val)
    if apply_curriculum:
        apply_curriculum_epochs(out, skip_epoch_keys=explicit)
    return out


# Back-compat alias used throughout training code
Config = TrainConfig

