"""Project and run-directory path resolution for active learning.

Run:
    Import only — ``from active_learning_pi.al.paths import repo_root, iteration_dir``."""
from __future__ import annotations

from pathlib import Path

from repo_paths import REPO_ROOT


def repo_root(start: Path | None = None) -> Path:
    """Return repository root (delegates to ``repo_paths.REPO_ROOT``)."""
    return REPO_ROOT


def gan_root(start: Path | None = None) -> Path:
    """Deprecated alias for :func:`repo_root`."""
    return repo_root(start)


def al_root(groot: Path | None = None) -> Path:
    return (groot or repo_root()) / "active_learning_pi"


def run_dir(cfg: dict, groot: Path | None = None) -> Path:
    root = al_root(groot) / "runs" / str(cfg["run_name"])
    root.mkdir(parents=True, exist_ok=True)
    return root


def iteration_dir(cfg: dict, iteration: int, groot: Path | None = None) -> Path:
    d = run_dir(cfg, groot) / f"iter_{iteration:04d}"
    d.mkdir(parents=True, exist_ok=True)
    return d
