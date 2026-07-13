"""Backward-compatible shim — import from ``repo_paths`` instead."""
from repo_paths import (  # noqa: F401
    ACTIVE_LEARNING_DIR,
    CONFIGS_DIR,
    DATA_MULTI_NORM,
    DATA_MULTI_NORM_ROBUST,
    DATA_MULTI_NORM_UNBOUNDED,
    DATASETS_DIR,
    EXPERIMENTS_DIR,
    HEATMAPS_DATA,
    LATENT_RUNS_DATA,
    PIPELINES_DIR,
    PROJECT_ROOT,
    REPO_ROOT,
    SCRAP_DIR,
    TOOLS_DIR,
    repo_path,
    setup_path,
)
