"""Build unbounded robust per-MHz dataset from raw ``data_multifreq_train``.

Run:
    python pipelines/normalize/build_train_norm_unbounded.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_BOOT = Path(__file__).resolve().parents[2]
if str(_REPO_BOOT) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOT))

from repo_paths import REPO_ROOT, setup_path

setup_path()

import pipelines.normalize.multifreq as norm  # noqa: E402


def main() -> None:
    norm.DATA_DIR = REPO_ROOT / "datasets" / "data_multifreq_train"
    norm.OUTPUT_DIR = REPO_ROOT / "datasets" / "data_multifreq_train_norm_unbounded"
    norm.USE_ROBUST_PER_MHZ = True
    norm.USE_UNBOUNDED_Z = True
    norm.APPEND = False
    norm.OVERWRITE_OUTPUT = True
    norm.MAX_K = 30
    norm.main()


if __name__ == "__main__":
    main()
