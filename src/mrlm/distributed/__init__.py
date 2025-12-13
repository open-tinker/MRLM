"""
Distributed training utilities for MRLM.

Supports FSDP, DDP, and multi-node training.
"""

from mrlm.distributed.setup import (
    init_distributed,
    cleanup_distributed,
    is_distributed,
    get_rank,
    get_world_size,
    get_local_rank,
)
from mrlm.distributed.fsdp import setup_fsdp_model, get_fsdp_config
from mrlm.distributed.ddp import setup_ddp_model
from mrlm.distributed.utils import (
    all_reduce_mean,
    barrier,
    broadcast,
    gather_tensors,
)

__all__ = [
    "init_distributed",
    "cleanup_distributed",
    "is_distributed",
    "get_rank",
    "get_world_size",
    "get_local_rank",
    "setup_fsdp_model",
    "get_fsdp_config",
    "setup_ddp_model",
    "all_reduce_mean",
    "barrier",
    "broadcast",
    "gather_tensors",
]
