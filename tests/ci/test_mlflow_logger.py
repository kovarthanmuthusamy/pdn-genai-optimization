"""tools/mlflow_log_run.py on a small synthetic run folder (written like thesis_code's s08_monitoring)."""
import json
import sys
from pathlib import Path

import pytest

mlflow = pytest.importorskip("mlflow")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from mlflow_log_run import log_run, read_metrics  # noqa: E402


@pytest.fixture()
def run_dir(tmp_path):
    d = tmp_path / "exp_demo"
    (d / "metrics" / "plots").mkdir(parents=True)
    (d / "checkpoints").mkdir()
    (d / "checkpoints" / "last_model.pt").write_bytes(b"weights")
    (d / "metrics" / "run_config.json").write_text(json.dumps({"latent_dim": 128, "lr": 1e-4, "phases": {"a": 1}}))
    (d / "metrics" / "run_meta.json").write_text(json.dumps({"git_sha": "abc123", "torch": "2.7.1"}))
    (d / "metrics" / "loss.csv").write_text("epoch,train_total_loss,val_total_loss\n1,3.0,2.5\n2,2.0,nan\n3,1.5,1.25\n")
    (d / "metrics" / "notes.csv").write_text("name,value\nx,1\n")         # no epoch column: skipped
    (d / "metrics" / "plots" / "curve.png").write_bytes(b"\x89PNG")
    return d


def test_reads_epoch_metrics_and_skips_nan(run_dir):
    rows = read_metrics(run_dir / "metrics")
    assert ("loss/val_total_loss", 1.25, 3) in rows
    assert not any(name == "loss/val_total_loss" and step == 2 for name, _, step in rows)
    assert not any(name.startswith("notes/") for name, _, _ in rows)


def test_logs_once(run_dir, tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    art = tmp_path / "artifacts"
    rid = log_run(run_dir, uri, "ci", artifact_root=art)
    assert log_run(run_dir, uri, "ci", artifact_root=art) == rid              # second call finds the run
    from mlflow.tracking import MlflowClient
    c = MlflowClient(tracking_uri=uri)
    run = c.get_run(rid)
    assert run.data.params["latent_dim"] == "128" and run.data.params["phases.a"] == "1"
    assert run.data.tags["meta.git_sha"] == "abc123" and run.data.tags["checkpoint_bytes"] == "7"
    assert [m.value for m in sorted(c.get_metric_history(rid, "loss/train_total_loss"), key=lambda m: m.step)] == [3.0, 2.0, 1.5]
    assert {a.path for a in c.list_artifacts(rid)} == {"plots", "run_record"}
    assert (art / "ci").is_dir()
    assert log_run(run_dir, uri, "ci", overwrite=True, artifact_root=art) != rid   # overwrite replaces the run
