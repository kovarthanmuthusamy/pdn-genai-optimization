"""Tests for Pydantic config validation (exp059_capacity_freq)."""
from pathlib import Path

import pytest
from pydantic import ValidationError

from experiments.exp059_capacity_freq.codes.config_schema import TrainConfig, train_config_from_yaml


def test_trainconfig_valid_dict():
    """Test that TrainConfig accepts a valid config dict without error."""
    cfg_dict = {
        "latent_dim": 128,
        "batch_size": 96,
        "learning_rate": 1.5e-5,
        "num_epochs": 500,
    }
    cfg = TrainConfig(**cfg_dict)
    assert cfg.latent_dim == 128
    assert cfg.batch_size == 96
    assert cfg.learning_rate == 1.5e-5


def test_trainconfig_invalid_type_raises():
    """Test that type mismatch (batch_size='string') raises ValidationError."""
    cfg_dict = {
        "batch_size": "not_a_number",
    }
    with pytest.raises(ValidationError):
        TrainConfig(**cfg_dict)


def test_trainconfig_loads_from_disk():
    """Test that we can load the real config.yaml from disk."""
    cfg_path = Path(__file__).resolve().parents[1] / "experiments" / "exp059_capacity_freq" / "config.yaml"
    assert cfg_path.is_file(), f"Config file not found at {cfg_path}"

    cfg = train_config_from_yaml(cfg_path)
    assert isinstance(cfg, TrainConfig)
    assert cfg.latent_dim > 0
    assert cfg.batch_size > 0
    assert cfg.num_epochs > 0
