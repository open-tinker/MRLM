"""
Distributed training setup and initialization.

Utilities for initializing and managing distributed training across multiple
GPUs and nodes.
"""

import os
from typing import Optional

import torch
import torch.distributed as dist


def init_distributed(
    backend: str = "nccl",
    init_method: Optional[str] = None,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    timeout_minutes: int = 30,
) -> None:
    """
    Initialize distributed training.

    This function should be called before any distributed operations.
    It automatically detects environment variables set by torchrun/slurm.

    Args:
        backend: Distributed backend ('nccl', 'gloo', 'mpi')
        init_method: Initialization method (e.g., 'env://', 'tcp://...')
        rank: Rank of current process (auto-detected if None)
        world_size: Total number of processes (auto-detected if None)
        timeout_minutes: Timeout for distributed operations in minutes

    Example:
        >>> # Launch with torchrun:
        >>> # torchrun --nproc_per_node=4 train.py
        >>> init_distributed()  # Auto-detects settings
    """
    # Check if already initialized
    if dist.is_initialized():
        print("Distributed training already initialized")
        return

    # Auto-detect rank and world_size from environment
    if rank is None:
        rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))

    if world_size is None:
        world_size = int(
            os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1"))
        )

    # Auto-detect init_method if not provided
    if init_method is None:
        if "MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ:
            init_method = "env://"
        else:
            # Single machine, use default
            init_method = "env://"
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "12355")

    # Set timeout
    timeout = torch.distributed.default_pg_timeout
    if timeout_minutes > 0:
        timeout = torch.distributed.timedelta(minutes=timeout_minutes)

    # Initialize process group
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timeout,
    )

    # Set device for current process
    if torch.cuda.is_available():
        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)

    print(
        f"Initialized distributed training: rank={rank}/{world_size}, "
        f"backend={backend}, init_method={init_method}"
    )


def cleanup_distributed() -> None:
    """
    Clean up distributed training.

    Should be called at the end of distributed training.
    """
    if dist.is_initialized():
        dist.destroy_process_group()
        print("Distributed training cleaned up")


def is_distributed() -> bool:
    """
    Check if distributed training is initialized.

    Returns:
        True if distributed training is active
    """
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """
    Get the rank of the current process.

    Returns:
        Rank (0 for main process, or if not distributed)
    """
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """
    Get the total number of processes.

    Returns:
        World size (1 if not distributed)
    """
    if is_distributed():
        return dist.get_world_size()
    return 1


def get_local_rank() -> int:
    """
    Get the local rank (rank within a single node).

    Returns:
        Local rank (0 if not distributed)
    """
    if is_distributed():
        # Try to get from environment
        local_rank = int(
            os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0"))
        )
        return local_rank
    return 0


def is_main_process() -> bool:
    """
    Check if this is the main process (rank 0).

    Returns:
        True if this is the main process
    """
    return get_rank() == 0


def setup_for_distributed(is_main: bool) -> None:
    """
    Disable printing for non-main processes.

    This helps reduce log clutter in multi-process training.

    Args:
        is_main: Whether this is the main process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_main or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def get_device() -> torch.device:
    """
    Get the appropriate device for the current process.

    Returns:
        torch.device for current process
    """
    if torch.cuda.is_available():
        if is_distributed():
            return torch.device(f"cuda:{get_local_rank()}")
        return torch.device("cuda")
    return torch.device("cpu")
