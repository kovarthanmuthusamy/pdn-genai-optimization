"""Multi-GPU (torchrun / DDP) helpers. Every function is a no-op in a single-process run.

Usage (import only):
init_distributed()            -> (local_rank, rank, world_size); NCCL, timeout NCCL_TIMEOUT_MINUTES (30)
cleanup_distributed()
get_rank() / get_world_size() / is_main_process()
barrier()
broadcast_float(value, device) rank 0's value on every rank
wrap_ddp(model, local_rank)   DistributedDataParallel (find_unused_parameters=True: the cross-frequency
                              and cycle decodes do not touch every parameter on every batch)
unwrap_model(model)           the underlying module (strips DDP / torch.compile wrappers)
model_state_dict(model)       state dict of the unwrapped module
"""
from __future__ import annotations

import datetime
import os

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
    if not distributed_enabled():
        return 0, 0, 1
    if not dist.is_initialized():
        # rank-0-only validation keeps the other rank waiting at a barrier; NCCL's default 10 min
        # timeout is too tight for that
        minutes = float(os.environ.get("NCCL_TIMEOUT_MINUTES", "30"))
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=minutes))
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
            return m


def wrap_ddp(model: torch.nn.Module, *, local_rank: int) -> torch.nn.Module:
    if get_world_size() <= 1:
        return model
    return DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)


def broadcast_float(value: float, *, device: str) -> float:
    if get_world_size() <= 1:
        return value
    t = torch.tensor([value], device=device, dtype=torch.float64)
    dist.broadcast(t, src=0)
    return float(t.item())


def model_state_dict(model: torch.nn.Module) -> dict:
    return unwrap_model(model).state_dict()
