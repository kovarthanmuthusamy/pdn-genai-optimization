"""Log a finished thesis_code training run to MLflow (parameters, per-epoch metrics, run record, plots).

What: thesis_code's training writes its own run record (s08_monitoring): metrics/run_config.json (the full
    configuration), metrics/run_meta.json (git commit, versions, device, config fingerprint), the per-epoch
    metric CSVs (loss.csv, epoch_timing.csv, latent_stats.csv, crossfreq_eval.csv, ...) and plots. This
    tool turns such a run folder into one MLflow run, after the fact, so thesis_code itself stays as the
    thesis documents it:
      params     every run_config.json value (MLflow keeps strings of up to 500 characters)
      tags       run_meta.json, plus the source folder and the checkpoint's size and SHA-256
      metrics    every numeric column of every metrics/*.csv with an "epoch" column, step = epoch,
                 named <csv stem>/<column> (e.g. loss/val_total_loss); NaN cells are skipped
      artifacts  the run record JSON files, the CSVs and metrics/plots/
    A run folder is logged once per experiment: a second call finds the run by its source tag and stops,
    unless OVERWRITE is set (then the old run is deleted first).
    The default store is SQLite (mlflow.db) with artifacts in mlartifacts/: MLflow 3 no longer accepts
    new runs in the plain ./mlruns file store.
Usage: edit the CONFIGURATION block, then  python tools/mlflow_log_run.py
    Browse with  mlflow ui --backend-store-uri sqlite:///mlflow.db  (from the repo root -> http://127.0.0.1:5000).
Config keys: RUN_DIR, TRACKING_URI, ARTIFACT_ROOT, EXPERIMENT, RUN_NAME, CHECKPOINT, LOG_CHECKPOINT, OVERWRITE.
Needs: pip install -e ".[tracking]"  (mlflow).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# =============================================================================
# CONFIGURATION — edit these before running
# =============================================================================
RUN_DIR = REPO / "experiments/exp096_k1k10_balanced30k_ep700"   # holds metrics/ (and checkpoints/)
TRACKING_URI = f"sqlite:///{REPO / 'mlflow.db'}"                # or a server, e.g. "http://host:5000"
ARTIFACT_ROOT = REPO / "mlartifacts"                            # artifacts of a new experiment (local store)
EXPERIMENT = "pdn-surrogate"
RUN_NAME: str | None = None                                     # None = the run folder's name
CHECKPOINT = "checkpoints/last_model.pt"                        # relative to RUN_DIR; size + SHA-256 as tags
LOG_CHECKPOINT = False                                          # True: also upload the checkpoint (~120 MB)
OVERWRITE = False                                               # True: replace an earlier log of RUN_DIR
# =============================================================================

PARAM_MAX = 500          # MLflow's limit on a parameter value
BATCH = 1000             # MLflow's limit on metrics per log_batch call


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v if isinstance(v, str) else json.dumps(v)
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_metrics(metrics_dir: Path) -> list[tuple[str, float, int]]:
    """(name, value, epoch) for every numeric cell of every CSV with an 'epoch' column."""
    rows = []
    for f in sorted(metrics_dir.glob("*.csv")):
        with open(f, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or "epoch" not in reader.fieldnames:
                continue
            for r in reader:
                try:
                    step = int(float(r["epoch"]))
                except (TypeError, ValueError):
                    continue
                for col, val in r.items():
                    if col == "epoch" or val in (None, ""):
                        continue
                    try:
                        x = float(val)
                    except ValueError:
                        continue
                    if math.isfinite(x):
                        rows.append((f"{f.stem}/{col}", x, step))
    return rows


def log_run(run_dir: Path, tracking_uri: str, experiment: str, run_name: str | None = None,
            checkpoint: str | None = CHECKPOINT, log_checkpoint: bool = False, overwrite: bool = False,
            artifact_root: Path | None = None) -> str:
    """Log ``run_dir`` as one MLflow run; returns the run id (an existing one when already logged)."""
    import mlflow
    from mlflow.entities import Metric, Param
    from mlflow.tracking import MlflowClient

    run_dir = Path(run_dir).resolve()
    metrics_dir = run_dir / "metrics"
    if not metrics_dir.is_dir():
        raise SystemExit(f"{metrics_dir} not found: RUN_DIR must be a thesis_code training folder")
    client = MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        loc = (Path(artifact_root) / experiment).resolve().as_uri() if artifact_root else None
        exp_id = client.create_experiment(experiment, artifact_location=loc)
    else:
        exp_id = exp.experiment_id
    found = client.search_runs([exp_id], filter_string=f"tags.source_run_dir = '{run_dir}'")
    if found and not overwrite:
        print(f"already logged as run {found[0].info.run_id} (set OVERWRITE to replace it)")
        return found[0].info.run_id
    for r in found:
        client.delete_run(r.info.run_id)

    cfg = json.loads((metrics_dir / "run_config.json").read_text()) if (metrics_dir / "run_config.json").is_file() else {}
    meta = json.loads((metrics_dir / "run_meta.json").read_text()) if (metrics_dir / "run_meta.json").is_file() else {}
    tags = {f"meta.{k}": str(v)[:5000] for k, v in _flat(meta).items()}
    tags.update({"source_run_dir": str(run_dir), "mlflow.runName": run_name or run_dir.name, "logged_by": Path(__file__).name})
    ck = run_dir / checkpoint if checkpoint else None
    if ck is not None and ck.is_file():
        tags.update({"checkpoint": checkpoint, "checkpoint_bytes": str(ck.stat().st_size), "checkpoint_sha256": _sha256(ck)})

    run = client.create_run(exp_id, tags=tags)
    rid = run.info.run_id
    params = [Param(k, str(v)[:PARAM_MAX]) for k, v in _flat(cfg).items()]
    metrics = [Metric(name, value, 0, step) for name, value, step in read_metrics(metrics_dir)]
    for i in range(0, len(params), 100):
        client.log_batch(rid, params=params[i:i + 100])
    for i in range(0, len(metrics), BATCH):
        client.log_batch(rid, metrics=metrics[i:i + BATCH])
    for f in sorted(metrics_dir.glob("*.json")) + sorted(metrics_dir.glob("*.csv")):
        client.log_artifact(rid, str(f), artifact_path="run_record")
    if (metrics_dir / "plots").is_dir():
        client.log_artifacts(rid, str(metrics_dir / "plots"), artifact_path="plots")
    if log_checkpoint and ck is not None and ck.is_file():
        client.log_artifact(rid, str(ck), artifact_path="checkpoint")
    client.set_terminated(rid)
    print(f"logged {run_dir.name}: {len(params)} params, {len(metrics)} metric values, "
          f"{len({m.key for m in metrics})} metric series -> run {rid} in '{experiment}' ({mlflow.__version__})")
    return rid


def main() -> None:
    log_run(Path(RUN_DIR), TRACKING_URI, EXPERIMENT, RUN_NAME, CHECKPOINT, LOG_CHECKPOINT, OVERWRITE, ARTIFACT_ROOT)


if __name__ == "__main__":
    sys.exit(main())
