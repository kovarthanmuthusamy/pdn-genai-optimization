"""Run record of a training run: metrics/run_meta.json, metrics/run_config.json, logs/train_events.jsonl.

What: run_meta.json holds the start / end time, the code version (git commit), the Python and
    PyTorch versions, the device and a fingerprint of the configuration; run_config.json the full
    configuration. train_events.jsonl is an event log, one JSON object per line: run_start, one
    "epoch" heartbeat per logged epoch (with the headline loss; every value is in metrics/loss.csv),
    the cross-frequency score, run_end.
Usage (import only):
    obs = RunObservability(exp_name=..., metrics_dir=..., log_dir=..., cfg=..., start_epoch=0,
                           world_size=1, device="cpu")
    obs.start(); obs.log_losses(epoch, train=..., val=...); obs.log_off_anchor(epoch, [], aggregate_score=s)
    obs.end(status="finished")
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_sha() -> str | None:
    """Short HEAD commit; None outside a git checkout."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], stderr=subprocess.DEVNULL,
                                      text=True).strip()
        return out or None
    except (OSError, subprocess.CalledProcessError):
        return None


def config_fingerprint(cfg: Mapping[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class JsonlEpochLogger:
    """Append-only events under logs/ (one JSON object per line)."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **fields: Any) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": utc_now_iso(), "event": event, **fields}, default=str) + "\n")


class RunObservability:
    def __init__(self, *, exp_name: str, metrics_dir: Path, log_dir: Path, cfg: Mapping[str, Any],
                 start_epoch: int = 0, world_size: int = 1, device: str = "cpu"):
        self.exp_name = exp_name
        self.metrics_dir = Path(metrics_dir)
        self.cfg = dict(cfg)
        self.start_epoch = int(start_epoch)
        self.world_size = int(world_size)
        self.device = device
        self.jsonl = JsonlEpochLogger(Path(log_dir) / "train_events.jsonl")

    def start(self) -> None:
        import torch
        meta = {"exp_name": self.exp_name, "experiment_dir": str(self.cfg.get("experiment_dir", "")),
                "started_at": utc_now_iso(), "git_sha": git_sha(), "python": sys.version.split()[0],
                "torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available()),
                "device": self.device, "world_size": self.world_size, "start_epoch": self.start_epoch,
                "resume_checkpoint": self.cfg.get("resume_checkpoint"),
                "config_fingerprint": config_fingerprint(self.cfg),
                "n_decap_types": self.cfg.get("n_decap_types"), "data_dir": self.cfg.get("data_dir")}
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        (self.metrics_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
        (self.metrics_dir / "run_config.json").write_text(json.dumps(self.cfg, indent=2, default=str) + "\n",
                                                          encoding="utf-8")
        self.jsonl.emit("run_start", exp=self.exp_name, start_epoch=self.start_epoch)

    def log_losses(self, step: int, train: Mapping[str, Any] | None = None,
                   val: Mapping[str, Any] | None = None) -> None:
        def headline(d):
            try:
                return float(d["total_loss"]) if d and "total_loss" in d else None
            except (TypeError, ValueError):
                return None
        total = headline(val) if headline(val) is not None else headline(train)
        self.jsonl.emit("epoch", epoch=int(step), n_metrics=len(train or {}) + len(val or {}),
                        val_ran=bool(val), total_loss=total, source="metrics/loss.csv")

    def log_off_anchor(self, step: int, rows: list, *, aggregate_score: float | None = None) -> None:
        self.jsonl.emit("off_anchor", epoch=int(step), n_rows=len(rows), aggregate_score=aggregate_score)

    def end(self, *, status: str = "finished", **fields: Any) -> None:
        self.jsonl.emit("run_end", status=status, **fields)
        meta_path = self.metrics_dir / "run_meta.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta.update({"ended_at": utc_now_iso(), "end_status": status, **fields})
            meta_path.write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
