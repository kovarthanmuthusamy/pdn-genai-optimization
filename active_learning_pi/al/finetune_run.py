"""Resolve AL fine-tune schedule from ``last_model.pt`` checkpoint epoch."""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import torch


def prepare_line_buffered_logging() -> None:
    """Avoid parent prints appearing after the training subprocess when stdout is redirected."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass


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


def _default_exp_dir(cfg: dict) -> str:
    return str(
        cfg.get("finetune", {}).get("experiment_dir")
        or cfg.get("experiment_dir")
        or "experiments/exp058_asymmetric_kl"
    )


def resolve_checkpoint_path(cfg: dict, groot: Path) -> Path:
    ft = cfg.get("finetune", {})
    exp = _default_exp_dir(cfg)
    rel = ft.get("checkpoint_path") or f"{exp}/checkpoints/last_model.pt"
    ckpt = groot / rel
    if not ckpt.is_file():
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
    exp = _default_exp_dir(cfg)
    base_rel = ft.get("config_path", f"{exp}/config_al_finetune.yaml")
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

    # Keep AL fine-tune KL behavior aligned with occ-only warm-up settings.
    occ_cfg_rel = (
        cfg.get("pre_al_occ_only_warmup", {}).get("config_path")
        or f"{exp}/config_occ_only_finetune.yaml"
    )
    occ_cfg_path = groot / occ_cfg_rel
    if occ_cfg_path.is_file():
        occ_cfg = _load_yaml_like(occ_cfg_path)
        for key in ("beta_final", "free_bits", "use_beta_annealing"):
            if key in occ_cfg and occ_cfg[key] is not None:
                data[key] = occ_cfg[key]

    for key in (
        "al_overlay_sample_weight",
        "layout_train_prob",
        "occ_only_encode_prob",
        "eval_use_occ_only_layout",
        "latent_distill_weight",
        "output_distill_weight",
        "latent_distill_logvar_weight",
        "heatmap_phys_p99_weight",
        "eval_off_anchor_mhz",
        "eval_off_anchor_mhz_weights",
        "eval_off_anchor_interval",
        "eval_off_anchor_max_batches",
        "al_finetune_early_stop",
        "al_finetune_early_stop_patience",
        "al_finetune_early_stop_min_delta",
        "al_finetune_early_stop_kind",
        "checkpoint_interval",
        "beta_final",
        "free_bits",
        "use_beta_annealing",
    ):
        if key in ft and ft[key] is not None:
            data[key] = ft[key]

    # Ensure eval matches AL scoring path.
    data["eval_use_occ_only_layout"] = True

    runtime = groot / exp / "config_al_finetune.runtime.yaml"
    _write_yaml_like(runtime, data)

    print(
        f"  Fine-tune schedule: resume {ckpt.name} @ epoch {start_ep} → {target_epochs} (+{extra})",
        flush=True,
    )
    print(
        "  KL: "
        f"beta={data.get('beta_final', '?')}  "
        f"free_bits={data.get('free_bits', '?')}  "
        f"use_beta_annealing={data.get('use_beta_annealing', '?')}",
        flush=True,
    )
    print(
        "  Asymmetric train (near AL path): "
        f"layout_p={data.get('layout_train_prob', '?')}  "
        f"occ_only={data.get('occ_only_encode_prob', '?')}  "
        f"latent_distill={data.get('latent_distill_weight', '?')}  "
        f"output_distill={data.get('output_distill_weight', '?')}",
        flush=True,
    )
    return runtime


def finetune_env(cfg: dict, groot: Path, *, use_overlay: bool) -> dict[str, str]:
    ft = cfg.get("finetune", {})
    exp_rel = ft.get("experiment_dir", cfg.get("experiment_dir"))
    exp_dir = groot / exp_rel
    runtime_cfg = write_runtime_finetune_config(cfg, groot)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["VAE_EXPERIMENT_DIR"] = str(exp_dir)
    env["VAE_CONFIG_PATH"] = str(runtime_cfg)
    env.setdefault("VAE_DATA_DIR", str(groot / cfg["data_dir"]))
    env["PYTHONPATH"] = str(groot) + os.pathsep + env.get("PYTHONPATH", "")
    if use_overlay:
        env.pop("VAE_SKIP_AL_OVERLAY", None)
    else:
        env["VAE_SKIP_AL_OVERLAY"] = "1"
    env["MLFLOW_CYCLE_ID"] = os.environ.get("MLFLOW_CYCLE_ID", str(uuid.uuid4()))
    return env
