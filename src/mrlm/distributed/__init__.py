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
from mrlm.distributed.gpu_utils import (
    get_available_gpus,
    get_gpu_info,
    select_gpus,
    set_visible_gpus,
    get_optimal_device,
    get_gpu_memory_usage,
    print_gpu_info,
    auto_select_gpus_for_model,
    setup_gpu_environment,
)

__all__ = [
    # Distributed setup
    "init_distributed",
    "cleanup_distributed",
    "is_distributed",
    "get_rank",
    "get_world_size",
    "get_local_rank",
    # Model setup
    "setup_fsdp_model",
    "get_fsdp_config",
    "setup_ddp_model",
    # Communication
    "all_reduce_mean",
    "barrier",
    "broadcast",
    "gather_tensors",
    # GPU utilities
    "get_available_gpus",
    "get_gpu_info",
    "select_gpus",
    "set_visible_gpus",
    "get_optimal_device",
    "get_gpu_memory_usage",
    "print_gpu_info",
    "auto_select_gpus_for_model",
    "setup_gpu_environment",
]
