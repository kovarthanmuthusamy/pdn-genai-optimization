"""DDP helpers for exp055 multi-GPU training."""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def distributed_enabled() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed() -> tuple[int, int, int]:
    """Return (local_rank, rank, world_size). No-op when not launched via torchrun."""
    if not distributed_enabled():
        return 0, 0, 1
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank, get_rank(), get_world_size()


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    m = model
    while True:
        if hasattr(m, "module"):
            m = m.module
        elif hasattr(m, "_orig_mod"):
            m = m._orig_mod
        else:
            break
    return m


def wrap_ddp(model: torch.nn.Module, *, local_rank: int) -> torch.nn.Module:
    if get_world_size() <= 1:
        return model
    return DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )


def broadcast_float(value: float, *, device: str) -> float:
    if get_world_size() <= 1:
        return value
    t = torch.tensor([value], device=device, dtype=torch.float64)
    dist.broadcast(t, src=0)
    return float(t.item())


def model_state_dict(model: torch.nn.Module) -> dict:
    return unwrap_model(model).state_dict()


def apply_ddp_config(c, *, world_size: int, local_rank: int) -> None:
    """Tune config for multi-GPU: device, LR scale, RAM cache, workers."""
    if world_size <= 1:
        return
    c.device = f"cuda:{local_rank}"
    if getattr(c, "cache_in_ram", False):
        print(
            f"[rank {get_rank()}] DDP: cache_in_ram disabled "
            f"(would duplicate host RAM × {world_size})",
            flush=True,
        )
        c.cache_in_ram = False
    if int(getattr(c, "num_workers", 0)) == 0:
        c.num_workers = 4
    base_batch = float(getattr(c, "ddp_base_batch_size", 160) or 160)
    global_batch = float(c.batch_size) * world_size
    if getattr(c, "ddp_linear_lr_scale", True) and base_batch > 0:
        scale = global_batch / base_batch
        c.learning_rate = float(c.learning_rate) * scale
        if is_main_process():
            print(
                f"DDP: world_size={world_size}  per_gpu_batch={c.batch_size}  "
                f"global_batch={int(global_batch)}  lr_scale={scale:.3f}  "
                f"lr={c.learning_rate:.2e}",
                flush=True,
            )


@contextmanager
def main_process_first():
    """Run block on rank 0 first, then let other ranks proceed (barrier sync)."""
    if is_main_process():
        yield
        barrier()
    else:
        barrier()
        yield
