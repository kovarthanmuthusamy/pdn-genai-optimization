"""Shared experiment config loading with repo-relative path resolution."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from repo_paths import REPO_ROOT, resolve_repo_path, resolve_repo_path


def load_experiment_config(path: Path) -> dict[str, Any]:
    """Parse experiment config.yaml (JSON object with optional ``#`` comment lines)."""
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(line)
    return json.loads("\n".join(lines))


def data_dir_from_config(
    config_path: Path,
    *,
    fallback: str | Path = "data_multi_norm_unbounded",
) -> Path:
    """Resolve ``data_dir`` from experiment config under ``REPO_ROOT``."""
    if config_path.is_file():
        cfg = load_experiment_config(config_path)
        data_dir = cfg.get("data_dir")
        if data_dir:
            return resolve_repo_path(data_dir)
    return resolve_repo_path(fallback)


def experiment_dir_from_config(config_path: Path) -> Path:
    """Resolve ``experiment_dir`` from config, else parent of config file."""
    if config_path.is_file():
        cfg = load_experiment_config(config_path)
        exp_dir = cfg.get("experiment_dir")
        if exp_dir:
            return resolve_repo_path(exp_dir)
    return config_path.resolve().parent


def norm_stats_path(data_dir: Path) -> Path:
    """Find normalization_stats.json for a resolved data directory."""
    primary = data_dir / "normalization_stats.json"
    if primary.is_file():
        return primary
    fallbacks = (
        REPO_ROOT / "data_multi_norm_unbounded" / "normalization_stats.json",
        REPO_ROOT / "datasets" / "data_norm" / "normalization_stats.json",
        REPO_ROOT / "data_multi_norm" / "normalization_stats.json",
        REPO_ROOT / "data_multi_norm_robust" / "normalization_stats.json",
    )
    for p in fallbacks:
        if p.is_file():
            return p
    tried = "\n  ".join(str(p) for p in (primary, *fallbacks))
    raise FileNotFoundError(f"normalization_stats.json not found. Tried:\n  {tried}")
