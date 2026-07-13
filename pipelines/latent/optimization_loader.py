"""Lazy loader for pipelines.latent.optimize."""
from __future__ import annotations

import importlib

_MOD = "pipelines.latent.optimize"


def load_optimization_module():
    return importlib.import_module(_MOD)


def resolve_run_dir(run_dir, repo_root):
    lo = load_optimization_module()
    if run_dir is None:
        return lo.resolve_run_dir(repo_root, None)
    from pathlib import Path
    path = Path(run_dir)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()
