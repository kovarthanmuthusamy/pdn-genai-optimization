"""Single source of truth for repository root and canonical paths.

All entry scripts should import from here instead of hard-coding ``parents[N]`` or
absolute paths like ``/home/ubuntu/genai_pdn``.

Usage::

    from repo_paths import REPO_ROOT, setup_path, repo_path
    setup_path()  # ensure repo root is on sys.path
    data = repo_path("datasets", "data_multifreq_train")
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Auto-detected from this file's location (works regardless of folder name).
REPO_ROOT = Path(__file__).resolve().parent

# Optional override via environment (e.g. CI or wrapper scripts).
_env_root = os.environ.get("GENAI_PDN_ROOT") or os.environ.get("REPO_ROOT")
if _env_root:
    REPO_ROOT = Path(_env_root).resolve()

# Alias used by some scrap/generation scripts.
PROJECT_ROOT = REPO_ROOT


def setup_path() -> Path:
    """Ensure repo root is on ``sys.path``; return ``REPO_ROOT``."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return REPO_ROOT


def repo_path(*parts: str) -> Path:
    """Build a path under the repository root."""
    return REPO_ROOT.joinpath(*parts)


# ── Canonical data / artifact locations (relative to REPO_ROOT) ──────────────
DATASETS_DIR = REPO_ROOT / "datasets"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
SCRAP_DIR = REPO_ROOT / "scrap"
PIPELINES_DIR = REPO_ROOT / "pipelines"
TOOLS_DIR = REPO_ROOT / "tools"
CONFIGS_DIR = REPO_ROOT / "configs"
ACTIVE_LEARNING_DIR = REPO_ROOT / "active_learning_pi"

HEATMAPS_DATA = REPO_ROOT / "data" / "heatmaps"
LATENT_RUNS_DATA = REPO_ROOT / "data" / "latent_runs"

DATA_MULTI_NORM_UNBOUNDED = REPO_ROOT / "data_multi_norm_unbounded"
DATA_MULTI_NORM_ROBUST = REPO_ROOT / "data_multi_norm_robust"
DATA_MULTI_NORM = REPO_ROOT / "data_multi_norm"

# Known stale absolute prefixes from older checkouts / machine layouts.
_STALE_PREFIXES: tuple[str, ...] = (
    "/home/ubuntu/gan/",
    "/home/ubuntu/genai_pdn/",
    "/home/ubuntu/GAN/",
)


def bootstrap_from(caller_file: str | Path, depth: int) -> Path:
    """Ensure repo root is on ``sys.path`` before ``import repo_paths``.

    Use at the top of scripts run as files from subdirectories::

        bootstrap_from(__file__, depth=2)  # scrap/generation/foo.py
        from repo_paths import REPO_ROOT, setup_path
        setup_path()
    """
    root = Path(caller_file).resolve().parents[depth]
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root


def strip_stale_prefix(path_str: str) -> str:
    """Remove known machine-specific repo prefixes, returning a repo-relative path."""
    s = path_str.replace("\\", "/")
    for prefix in _STALE_PREFIXES:
        if s.startswith(prefix):
            return s[len(prefix) :].lstrip("/")
    return s.lstrip("/")


def resolve_repo_path(path: str | Path, *, must_exist: bool = False) -> Path:
    """Resolve a config/script path under ``REPO_ROOT``.

    - Relative paths → ``REPO_ROOT / path``
    - Stale absolutes (``/home/ubuntu/gan/...``) → remapped under ``REPO_ROOT``
    - Valid existing absolutes → returned as-is
    """
    if path is None or (isinstance(path, str) and not str(path).strip()):
        raise ValueError("path must be non-empty")

    raw = Path(path)
    if raw.is_absolute():
        if raw.exists():
            return raw.resolve()
        remapped = REPO_ROOT / strip_stale_prefix(str(raw))
        if remapped.exists() or not must_exist:
            return remapped.resolve()
        # Last resort: match by final path component under repo
        name = raw.name
        for candidate in REPO_ROOT.rglob(name):
            if candidate.is_dir() and raw.name == name:
                parent_match = strip_stale_prefix(str(raw.parent))
                if str(candidate).endswith(parent_match) or parent_match.endswith(candidate.name):
                    return candidate.resolve()
        return remapped.resolve()

    return (REPO_ROOT / raw).resolve()


def resolve_config_paths(cfg: dict, keys: tuple[str, ...] = ("data_dir", "experiment_dir", "resume_checkpoint")) -> dict:
    """Return a shallow copy of *cfg* with selected path keys resolved under ``REPO_ROOT``."""
    out = dict(cfg)
    for key in keys:
        val = out.get(key)
        if isinstance(val, str) and val.strip():
            out[key] = str(resolve_repo_path(val))
    return out
