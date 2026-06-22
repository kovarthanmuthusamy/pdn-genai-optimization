"""Project and run-directory path resolution for active learning.

Run:
    Import only — ``from active_learning_pi.al.paths import gan_root, iteration_dir``."""
from __future__ import annotations

from pathlib import Path


def gan_root(start: Path | None = None) -> Path:
    if start is None:
        start = Path(__file__).resolve()
    for p in [start, *start.parents]:
        if (p / "active_learning_pi").is_dir() and (p / "scrap").is_dir():
            return p
    raise RuntimeError("Could not locate gan project root (need active_learning_pi/ and scrap/)")


def al_root(groot: Path | None = None) -> Path:
    return (groot or gan_root()) / "active_learning_pi"


def run_dir(cfg: dict, groot: Path | None = None) -> Path:
    root = al_root(groot) / "runs" / str(cfg["run_name"])
    root.mkdir(parents=True, exist_ok=True)
    return root


def iteration_dir(cfg: dict, iteration: int, groot: Path | None = None) -> Path:
    d = run_dir(cfg, groot) / f"iter_{iteration:04d}"
    d.mkdir(parents=True, exist_ok=True)
    return d
