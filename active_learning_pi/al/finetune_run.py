"""Resolve exp057 AL fine-tune schedule from ``last_model.pt`` checkpoint epoch."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def _load_yaml_like(path: Path) -> dict[str, Any]:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(line)
    return json.loads("\n".join(lines))


def _write_yaml_like(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def resolve_checkpoint_path(cfg: dict, groot: Path) -> Path:
    ft = cfg.get("finetune", {})
    rel = ft.get("checkpoint_path") or "experiments/exp057_structured_graph/checkpoints/last_model.pt"
    ckpt = groot / rel
    if not ckpt.is_file():
        exp = ft.get("experiment_dir", cfg.get("experiment_dir", "experiments/exp057_structured_graph"))
        fallback = groot / exp / "checkpoints" / "last_model.pt"
        if fallback.is_file():
            return fallback
        raise FileNotFoundError(f"Fine-tune checkpoint not found: {ckpt}")
    return ckpt


def checkpoint_epoch(ckpt_path: Path) -> int:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return int(ck.get("epoch", 0))


def write_runtime_finetune_config(cfg: dict, groot: Path) -> Path:
    """Merge base AL finetune yaml with resume epoch + ``last_model.pt``."""
    ft = cfg.get("finetune", {})
    base_rel = ft.get("config_path", "experiments/exp057_structured_graph/config_al_finetune.yaml")
    base_path = groot / base_rel
    if not base_path.is_file():
        raise FileNotFoundError(f"Fine-tune config not found: {base_path}")

    ckpt = resolve_checkpoint_path(cfg, groot)
    start_ep = checkpoint_epoch(ckpt)
    extra = int(ft.get("extra_epochs", 50))
    target_epochs = start_ep + extra

    data = _load_yaml_like(base_path)
    data["resume_checkpoint"] = str(ckpt.relative_to(groot)).replace("\\", "/")
    if not Path(data["resume_checkpoint"]).is_file():
        data["resume_checkpoint"] = str(ckpt)
    data["num_epochs"] = target_epochs
    data.setdefault("reset_lr_on_resume", True)

    exp_rel = ft.get("experiment_dir", cfg.get("experiment_dir"))
    runtime = groot / exp_rel / "config_al_finetune.runtime.yaml"
    _write_yaml_like(runtime, data)

    print(f"  Fine-tune schedule: resume {ckpt.name} @ epoch {start_ep} → {target_epochs} (+{extra})")
    return runtime


def finetune_env(cfg: dict, groot: Path, *, use_overlay: bool) -> dict[str, str]:
    import os

    ft = cfg.get("finetune", {})
    exp_rel = ft.get("experiment_dir", cfg.get("experiment_dir"))
    exp_dir = groot / exp_rel
    runtime_cfg = write_runtime_finetune_config(cfg, groot)

    env = os.environ.copy()
    env["VAE_EXPERIMENT_DIR"] = str(exp_dir)
    env["VAE_CONFIG_PATH"] = str(runtime_cfg)
    env.setdefault("VAE_DATA_DIR", str(groot / cfg["data_dir"]))
    env["PYTHONPATH"] = str(groot) + os.pathsep + env.get("PYTHONPATH", "")
    if use_overlay:
        env.pop("VAE_SKIP_AL_OVERLAY", None)
    else:
        env["VAE_SKIP_AL_OVERLAY"] = "1"
    return env
