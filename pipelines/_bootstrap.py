"""Bootstrap repo root for all ``pipelines.*`` modules."""
from __future__ import annotations

from repo_paths import REPO_ROOT, repo_path, setup_path

setup_path()

__all__ = ["REPO_ROOT", "repo_path"]
