"""Training configuration: the schema (``TrainConfig``) and the config.yaml reader.

What: ``config.yaml`` is JSON with full-line ``#`` comments (not YAML). It must set every field of
    ``TrainConfig`` that has no default, and nothing else: an unknown key is an error. Phase starts
    are absolute 0-based epochs. The thesis model's file is common/inputs/config.yaml.
    ``data_dir`` is always settings.DATA_DIR/normalized (the value in the file is replaced), and
    ``experiment_dir`` (checkpoints, metrics, logs) defaults to OUTPUT_DIR/s07_training.
    ``background_value`` and the two map clip bounds are not configurable: the trainer reads them
    from the dataset's normalization_stats.json and stores them here only so checkpoints record them.
Usage (import only):
    load_train_config(paths) -> TrainConfig from config files applied in order
    load_experiment_config_dict(path) -> raw dict of a config file
    vae_model_kwargs(cfg) -> MultiInputVAE constructor kwargs from a config dict or TrainConfig
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field

from thesis_code.common import settings
from thesis_code.common.board import INPUTS

CONFIG_FILE = INPUTS / "config.yaml"

# Set from normalization_stats.json at start-up; config.yaml may not contain them.
FROM_DATASET = ("background_value", "heatmap_z_clip_min", "heatmap_z_clip_max")


def load_experiment_config_dict(path: Path) -> dict[str, Any]:
    """Parse a config.yaml file: a JSON object with optional full-line ``#`` comments."""
    lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    return json.loads("\n".join(lines)) if lines else {}


def vae_model_kwargs(cfg: dict[str, Any] | Any) -> dict[str, Any]:
    """Constructor kwargs for ``model.MultiInputVAE`` from a config dict or TrainConfig."""
    get = (lambda k, d=None: cfg.get(k, d)) if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    return {
        "latent_dim": int(get("latent_dim", 128)),
        "cond_dim": int(get("cond_dim", 32)),
        "freq_fourier_features": int(get("freq_fourier_features", 48)),
        "n_occ_classes": int(get("n_decap_types", 3)),
        "k_max": int(get("k_max", 10)),
        "modality_dropout": float(get("modality_dropout", 0.0)),
    }


class TrainConfig(BaseModel):
    """Every setting the training code reads. Fields without a default must be in config.yaml;
    see config.yaml for the values and what each one does."""

    # extra="allow" only for runtime attributes the trainer attaches (``_norm_stats``); unknown
    # config.yaml keys are rejected by ``load_train_config`` before they get here.
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True, validate_assignment=True)

    # ── Paths, devices, reproducibility ──
    data_dir: str
    experiment_dir: str = Field(default_factory=lambda: str(Path(settings.OUTPUT_DIR) / "s07_training"))
    device: str = Field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    seed: int                                 # torch/numpy/random, offset by DDP rank
    # Seed of the train/val LAYOUT split only. Pinned apart from `seed` because it defines the
    # held-out designs; anything that evaluates a checkpoint must use the same value.
    split_seed: int

    # ── Model ──
    latent_dim: int
    cond_dim: int
    freq_fourier_features: int
    n_decap_types: int
    k_max: int                                # largest decap budget; K embedding covers 0..k_max
    modality_dropout: float

    # ── Data ──
    batch_size: int                           # per GPU
    val_batch_size: int
    train_split: float
    train_samples_per_epoch: int              # draws per epoch (with replacement), split over GPUs
    num_workers: int
    persistent_workers: bool
    prefetch_factor: int

    # ── Optimisation (AdamW, weight decay 1e-4; ReduceLROnPlateau on the validation loss) ──
    num_epochs: int
    learning_rate: float
    lr_min: float
    lr_patience: int
    lr_factor: float
    reset_lr_on_resume: bool
    training_grad_clip_norm: float
    training_skip_nonfinite_batches: bool
    training_logvar_clamp_min: float
    training_logvar_clamp_max: float
    training_recon_z_soft_floor: float
    training_recon_z_cap_mult: float          # heatmap output capped at the dataset z_max x this
    training_loss_finite_cap: float

    # ── Checkpoints, validation, logging ──
    checkpoint_interval: int                  # validate + checkpoint every N epochs (and epoch 1, last)
    resume_checkpoint: int | str | None = None  # epoch number, path, "last" or "latest"
    epoch_print_interval: int
    crossfreq_eval_interval: int              # on validated epochs; picks best_off_anchor_model.pt
    crossfreq_eval_max_batches: int

    # ── Set from normalization_stats.json at start-up (FROM_DATASET) ──
    background_value: float | None = None     # zero point of the peak-location weighting
    heatmap_z_clip_min: float | None = None
    heatmap_z_clip_max: float | None = None

    # ── Loss weights: reconstruction ──
    heatmap_weight: float
    occupancy_weight: float
    impedance_weight: float
    occupancy_focal_gamma: float
    occ_k_consistency_weight: float

    # heatmap loss = pearson + level + region overlap + gradient field (losses/heatmap.py)
    heatmap_pearson_weight: float
    heatmap_level_weight: float
    heatmap_softoverlap_weight: float
    heatmap_softoverlap_temp: float
    heatmap_grad_vector_weight: float
    heatmap_grad_direction_weight: float
    heatmap_grad_direction_min_mag: float
    heatmap_grad_huber_delta: float
    heatmap_peak_loc_weight: float
    heatmap_peak_loc_temperature: float

    # impedance spectrum loss (losses/impedance.py)
    impedance_deriv_weight: float
    impedance_topk_k: int
    impedance_topk_weight: float
    impedance_under_penalty: float
    impedance_freq_weight_alpha: float
    impedance_peak_index_weight: float
    impedance_num_peaks: int

    # ── Loss weights: latent ──
    beta_final: float                         # KL weight (constant)
    free_bits: float
    mu_hinge_threshold: float
    mu_hinge_weight: float
    mu_bias_weight: float
    per_expert_kl_weight: float
    sigma_reg_weight: float
    sigma_reg_target: float

    # ── Loss weights: cross-frequency, anti-copy, cycle (train only) ──
    cross_freq_weight: float
    anticopy_weight: float
    full_encode_cycle_consistency_weight: float

    # ── Heatmap-focus phase: replaces the weights above from heatmap_focus_start_epoch ──
    heatmap_focus_heatmap_weight: float
    heatmap_focus_impedance_weight: float
    heatmap_focus_cross_freq_weight: float
    heatmap_focus_modality_dropout: float

    # ── Curriculum (absolute epochs, 0-based: a phase is on when epoch >= its start) ──
    penalty_warmup_epochs: int                # impedance top-k / peak terms ramp 0 -> 1 over these
    cross_freq_start_epoch: int
    anticopy_start_epoch: int
    modality_dropout_protect_heatmap_epoch: int
    heatmap_focus_start_epoch: int
    impedance_peak_start_epoch: int
    impedance_peak_ramp_epochs: int

    def is_cuda(self) -> bool:
        return "cuda" in self.device

    def to_checkpoint_dict(self) -> dict[str, Any]:
        """JSON-serialisable snapshot of the declared fields (runtime extras excluded)."""
        return self.model_dump(mode="json", exclude=set(self.__pydantic_extra__ or {}))


def load_train_config(paths: list[Path]) -> TrainConfig:
    """TrainConfig from config.yaml files applied in order (a later file overrides keys).

    Raises on any key that is not a ``TrainConfig`` field, on a FROM_DATASET key, and on a
    missing required field.
    """
    data: dict[str, Any] = {}
    for path in paths:
        part = {k: v for k, v in load_experiment_config_dict(path).items() if not k.startswith("_")}
        unknown = sorted(set(part) - set(TrainConfig.model_fields))
        if unknown:
            raise ValueError(f"{path}: unknown config keys (not read by any code): {unknown}")
        fixed = sorted(set(part) & set(FROM_DATASET))
        if fixed:
            raise ValueError(f"{path}: {fixed} come from normalization_stats.json, not config.yaml")
        data.update(part)
        print(f"Loaded config from {path} ({len(part)} keys)")
    data["data_dir"] = str(Path(settings.DATA_DIR) / "normalized")
    return TrainConfig(**data)
