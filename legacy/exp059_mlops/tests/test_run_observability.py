"""Tests for Observability 1.5 (run_meta, JSONL, MLflow facade)."""
from __future__ import annotations

import json
import os
from pathlib import Path

from src_vae.others.run_observability import (
    JsonlEpochLogger,
    RunObservability,
    config_fingerprint,
    default_run_tags,
    mlflow_param_dict,
    write_run_meta,
)


def test_mlflow_param_dict_caps_and_prefers_keys():
    cfg = {f"k{i}": i for i in range(200)}
    cfg.update({"latent_dim": 42, "n_decap_types": 2, "batch_size": 16})
    params = mlflow_param_dict(cfg)
    assert len(params) <= 100
    assert params["latent_dim"] == "42"
    assert params["n_decap_types"] == "2"


def test_default_run_tags_include_cycle_and_types(monkeypatch):
    monkeypatch.setenv("MLFLOW_CYCLE_ID", "cycle-xyz")
    tags = default_run_tags(
        exp_name="exp060_multitype_occ",
        cfg={"n_decap_types": 3, "resume_checkpoint": None},
        start_epoch=10,
        world_size=2,
        git="abc123",
    )
    assert tags["exp"] == "exp060_multitype_occ"
    assert tags["n_decap_types"] == "3"
    assert tags["cycle_id"] == "cycle-xyz"
    assert tags["git_sha"] == "abc123"
    assert tags["start_epoch"] == "10"
    assert tags["resume"] == "0"


def test_write_run_meta_and_jsonl(tmp_path: Path):
    metrics = tmp_path / "metrics"
    logs = tmp_path / "logs"
    cfg = {
        "latent_dim": 8,
        "n_decap_types": 2,
        "experiment_dir": str(tmp_path),
        "data_dir": "/data",
    }
    meta = write_run_meta(
        metrics / "run_meta.json",
        exp_name="exp060_multitype_occ",
        experiment_dir=tmp_path,
        cfg=cfg,
        start_epoch=0,
        world_size=1,
        device="cpu",
        mlflow_run_id=None,
    )
    assert meta["config_fingerprint"] == config_fingerprint(cfg)
    assert (metrics / "run_meta.json").is_file()

    jl = JsonlEpochLogger(logs / "train_events.jsonl")
    jl.emit("epoch", epoch=1, train={"train/total_loss": 1.0})
    rows = (logs / "train_events.jsonl").read_text().strip().splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["event"] == "epoch"


def test_run_observability_jsonl_and_meta_even_without_mlflow(tmp_path: Path, monkeypatch):
    """If MLflow import/tracking fails, meta + JSONL still work."""
    metrics = tmp_path / "metrics"
    logs = tmp_path / "logs"
    metrics.mkdir()
    logs.mkdir()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/ml.db")

    cfg = {
        "latent_dim": 42,
        "batch_size": 4,
        "n_decap_types": 2,
        "experiment_dir": str(tmp_path),
        "data_dir": "/tmp/data",
    }
    obs = RunObservability(
        exp_name="exp060_multitype_occ",
        metrics_dir=metrics,
        log_dir=logs,
        cfg=cfg,
        start_epoch=0,
        world_size=1,
        device="cpu",
        enabled=True,
    )
    obs.start()
    obs.log_losses(1, train={"total_loss": 2.5}, val={"total_loss": 2.0})
    obs.log_off_anchor(
        1,
        [
            {
                "kind": "encode_cross",
                "mhz": 100.0,
                "hm_fg_mse_mean": 0.5,
                "pearson_fg_mean": 0.9,
                "peak_loc_err_mean": 0.1,
            }
        ],
        aggregate_score=0.5,
    )
    obs.end(status="finished")

    meta = json.loads((metrics / "run_meta.json").read_text())
    assert meta["exp_name"] == "exp060_multitype_occ"
    assert meta["n_decap_types"] == 2
    assert meta.get("ended_at")
    assert (metrics / "run_config.json").is_file()

    events = [
        json.loads(line)["event"]
        for line in (logs / "train_events.jsonl").read_text().strip().splitlines()
    ]
    assert events == ["run_start", "epoch", "off_anchor", "run_end"]

    # Off-anchor metric keys should be present in the JSONL payload
    oa = json.loads((logs / "train_events.jsonl").read_text().strip().splitlines()[2])
    assert "off_anchor/encode_cross/100MHz/pearson" in oa["metrics"]
