#!/usr/bin/env python3
"""Active-learning PI pipeline for exp057.

Run:
    python pipelines/active_learning/run.py

Edit CONFIG below, then run the command above.

``full`` = all 7 steps: generate → infer → select → ECAD → ingest → overlay → fine-tune.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from repo_paths import setup_path  # noqa: E402

setup_path()

from active_learning_pi.al.pipeline import (  # noqa: E402
    bump_iteration,
    cmd_generate,
    cmd_infer,
    load_config,
    main_from_config,
)

# =============================================================================
# CONFIGURATION — edit these before running: python pipelines/active_learning/run.py
# =============================================================================

COMMAND = "full"
# full | cycle | generate | infer | select-bad | build-peb | simulate | ingest |
# normalize | evaluate | build-overlay | finetune | finetune-hint

CONFIG_PATH = "active_learning_pi/config/exp057.json"  # None → default.json

ITERATION = None  # force iteration index; None = latest from disk

SKIP_SIMULATE = False
SKIP_INGEST = False
PROPOSE_ONLY = False  # True → generate + infer only (no ECAD / no fine-tune)

# =============================================================================


def main() -> int:
    cfg = load_config(CONFIG_PATH)
    groot = Path(cfg.get("repo_root") or cfg["gan_root"])

    if PROPOSE_ONLY:
        it = bump_iteration(cfg, groot)
        cmd_generate(cfg, groot, it)
        cmd_infer(cfg, groot, it)
        return 0

    return main_from_config(
        command=COMMAND,
        config_path=CONFIG_PATH,
        iteration=ITERATION,
        skip_simulate=SKIP_SIMULATE,
        skip_ingest=SKIP_INGEST,
    )


if __name__ == "__main__":
    raise SystemExit(main())
