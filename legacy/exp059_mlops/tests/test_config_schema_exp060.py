"""Tests for Pydantic config validation (exp060_multitype_occ)."""
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from experiments.exp060_multitype_occ.codes.config_schema import (
    TrainConfig,
    apply_curriculum_epochs,
    train_config_from_yaml,
    vae_model_kwargs,
)
from experiments.exp060_multitype_occ.codes.occupancy_types import (
    n_occ_classes,
    type_ids_to_onehot,
)


def test_trainconfig_valid_dict():
    cfg = TrainConfig(
        latent_dim=128,
        batch_size=96,
        learning_rate=1.5e-5,
        num_epochs=500,
        n_decap_types=3,
    )
    assert cfg.latent_dim == 128
    assert cfg.batch_size == 96
    assert cfg.learning_rate == 1.5e-5
    assert cfg.n_decap_types == 3


def test_trainconfig_invalid_type_raises():
    with pytest.raises(ValidationError):
        TrainConfig(batch_size="not_a_number")


def test_trainconfig_loads_from_disk():
    cfg_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "exp060_multitype_occ"
        / "config.yaml"
    )
    assert cfg_path.is_file(), f"Config file not found at {cfg_path}"

    cfg = train_config_from_yaml(cfg_path)
    assert isinstance(cfg, TrainConfig)
    assert cfg.latent_dim > 0
    assert cfg.batch_size > 0
    assert cfg.num_epochs > 0
    assert cfg.n_decap_types >= 1
    # Curriculum maps frac → epoch unless yaml set an explicit *_epoch
    assert cfg.cross_freq_start_epoch >= 0
    assert cfg.beta_end_epoch > 0


def test_n_decap_types_maps_to_vae_kwargs():
    cfg = TrainConfig(n_decap_types=4)
    kwargs = vae_model_kwargs(cfg)
    assert kwargs["n_occ_classes"] == 4
    assert n_occ_classes(cfg.n_decap_types) == 4


def test_curriculum_skip_explicit_epoch():
    cfg = TrainConfig(num_epochs=100, beta_end_frac=0.1, beta_end_epoch=999)
    apply_curriculum_epochs(cfg, skip_epoch_keys={"beta_end_epoch"})
    assert cfg.beta_end_epoch == 999
    apply_curriculum_epochs(cfg, skip_epoch_keys=set())
    assert cfg.beta_end_epoch == 10


def test_occupancy_type_ids_to_onehot():
    codes = torch.tensor([0, 1, 2, 0] + [0] * 48)  # empty, type1, type2, empty...
    oh = type_ids_to_onehot(codes, n_decap_types=2)
    assert oh.shape == (52, 2)
    assert oh[0].sum() == 0
    assert torch.allclose(oh[1], torch.tensor([1.0, 0.0]))
    assert torch.allclose(oh[2], torch.tensor([0.0, 1.0]))
