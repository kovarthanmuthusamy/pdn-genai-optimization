#!/usr/bin/env python3
"""Pre-AL occupancy-only warm-up for exp057 (no overlay).

Trains decode-from-layout-only to match AL candidate scoring, then run AL:

    python pipelines/active_learning/finetune_occ_only_exp057.py
    python pipelines/active_learning/run.py   # COMMAND=propose or full

Edit CONFIG below.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from repo_paths import setup_path  # noqa: E402

setup_path()

from active_learning_pi.al.finetune_run import (  # noqa: E402
    checkpoint_epoch,
    prepare_line_buffered_logging,
    resolve_checkpoint_path,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_PATH = "active_learning_pi/config/exp057.json"
OCC_ONLY_CONFIG = "experiments/exp057_structured_graph/config_occ_only_finetune.yaml"
EXTRA_EPOCHS = 100

# =============================================================================


def _load_yaml_like(path: Path) -> dict:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(line)
    return json.loads("\n".join(lines))


def _write_yaml_like(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    prepare_line_buffered_logging()
    from active_learning_pi.al.config import load_config

    cfg = load_config(CONFIG_PATH)
    groot = Path(cfg["repo_root"])
    warmup = cfg.get("pre_al_occ_only_warmup", {})

    base_rel = warmup.get("config_path", OCC_ONLY_CONFIG)
    base_path = groot / base_rel
    if not base_path.is_file():
        print(f"Missing occ-only config: {base_path}")
        return 1

    extra = int(warmup.get("extra_epochs", EXTRA_EPOCHS))
    ckpt_rel = warmup.get("checkpoint_path") or cfg.get("checkpoint_path")
    ckpt = groot / ckpt_rel
    if not ckpt.is_file():
        ckpt = resolve_checkpoint_path(cfg, groot)

    start_ep = checkpoint_epoch(ckpt)
    target_epochs = start_ep + extra

    data = _load_yaml_like(base_path)
    data["resume_checkpoint"] = str(ckpt.relative_to(groot)).replace("\\", "/")
    if not Path(data["resume_checkpoint"]).is_file():
        data["resume_checkpoint"] = str(ckpt)
    data["num_epochs"] = target_epochs
    data.setdefault("reset_lr_on_resume", True)

    exp_rel = cfg.get("experiment_dir", "experiments/exp057_structured_graph")
    runtime = groot / exp_rel / "config_occ_only_finetune.runtime.yaml"
    _write_yaml_like(runtime, data)

    train_script = groot / exp_rel / "codes" / "train_vae_simple.py"
    if not train_script.is_file():
        print(f"Missing: {train_script}")
        return 1

    print("\n=== Pre-AL occupancy-only warm-up ===", flush=True)
    print(f"  checkpoint: {ckpt}", flush=True)
    print(f"  schedule: epoch {start_ep} → {target_epochs} (+{extra})", flush=True)
    print(f"  layout_train_prob=1.0  occ_only_encode_prob=1.0  (no overlay)", flush=True)
    print(
        f"  KL: beta={data.get('beta_final', '?')}  free_bits={data.get('free_bits', '?')}  "
        f"use_beta_annealing={data.get('use_beta_annealing', '?')}",
        flush=True,
    )
    print(f"  runtime config: {runtime}", flush=True)
    print("  After this completes, run AL (infer/score) with last_model.pt.", flush=True)

    import os

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["VAE_EXPERIMENT_DIR"] = str(groot / exp_rel)
    env["VAE_CONFIG_PATH"] = str(runtime)
    env.setdefault("VAE_DATA_DIR", str(groot / cfg["data_dir"]))
    env["VAE_SKIP_AL_OVERLAY"] = "1"
    env["PYTHONPATH"] = str(groot) + os.pathsep + env.get("PYTHONPATH", "")

    return subprocess.call([sys.executable, str(train_script)], cwd=str(groot), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
