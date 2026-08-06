#!/usr/bin/env python3
"""Fine-tune exp057 after active-learning (Option B: heatmap-only overlay).

Run:
    python pipelines/active_learning/finetune_exp057.py

Edit CONFIG below. Usually invoked automatically by ``run.py`` COMMAND=full.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from repo_paths import setup_path  # noqa: E402

setup_path()

from active_learning_pi.al.build_overlay import build_overlay_from_iterations  # noqa: E402
from active_learning_pi.al.config import load_config  # noqa: E402
from active_learning_pi.al.finetune_run import (  # noqa: E402
    finetune_env,
    prepare_line_buffered_logging,
    resolve_checkpoint_path,
)
from src_vae.others.multifreq_layout_store import load_manifest_rows  # noqa: E402

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_PATH = "active_learning_pi/config/exp057.json"
SKIP_OVERLAY_BUILD = False  # True → use existing overlay on disk
TRAIN_ONLY = False          # True → no AL overlay (base dataset only)

# =============================================================================


def main() -> int:
    prepare_line_buffered_logging()
    cfg = load_config(CONFIG_PATH)
    groot = Path(cfg["repo_root"])
    ft = cfg.get("finetune", {})

    if not ft.get("enabled"):
        print("finetune.enabled is false in AL config.")
        return 1

    overlay_rel = ft.get("overlay_data_dir") or cfg.get("overlay_data_dir")
    overlay_root = groot / overlay_rel if overlay_rel else None
    use_overlay = not TRAIN_ONLY

    if use_overlay and not SKIP_OVERLAY_BUILD:
        print("\n=== Build AL overlay dataset (Option B) ===", flush=True)
        report = build_overlay_from_iterations(cfg, groot)
        print(
            f"  added={report.get('added')} skipped={report.get('skipped_existing')}",
            flush=True,
        )
        if report.get("errors"):
            return 1

    n_overlay = len(load_manifest_rows(overlay_root)) if overlay_root and overlay_root.is_dir() else 0
    if use_overlay and n_overlay == 0:
        print("No AL overlay samples. Set COMMAND=full in pipelines/active_learning/run.py")
        return 1

    exp_rel = ft["experiment_dir"]
    train_script = groot / exp_rel / "codes" / "train_vae_simple.py"
    if not train_script.is_file():
        print(f"Missing: {train_script}")
        return 1

    ckpt = resolve_checkpoint_path(cfg, groot)
    print("\n=== Launch exp057 AL fine-tune ===", flush=True)
    print(f"  checkpoint: {ckpt}", flush=True)
    if use_overlay:
        print(f"  overlay samples: {n_overlay}", flush=True)

    env = finetune_env(cfg, groot, use_overlay=use_overlay)
    return subprocess.call([sys.executable, str(train_script)], cwd=str(groot), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
