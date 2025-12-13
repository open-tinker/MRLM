"""
DDP (Distributed Data Parallel) utilities.

DDP replicates the model across GPUs and synchronizes gradients.
More memory-intensive than FSDP but simpler and faster for smaller models.
"""

from typing import Optional, List

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from mrlm.distributed.setup import get_local_rank, is_distributed


def setup_ddp_model(
    model: nn.Module,
    device_ids: Optional[List[int]] = None,
    output_device: Optional[int] = None,
    find_unused_parameters: bool = False,
    gradient_as_bucket_view: bool = True,
    static_graph: bool = False,
) -> DDP:
    """
    Wrap model with DDP (Distributed Data Parallel).

    Args:
        model: Model to wrap
        device_ids: List of device IDs (auto-detected if None)
        output_device: Output device ID (auto-detected if None)
        find_unused_parameters: Whether to find unused parameters
        gradient_as_bucket_view: Memory optimization for gradients
        static_graph: Whether computation graph is static (enables optimizations)

    Returns:
        DDP-wrapped model

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> model = model.to("cuda")
        >>> ddp_model = setup_ddp_model(model)
    """
    if not is_distributed():
        print("Warning: DDP requested but distributed training not initialized")
        return model

    # Auto-detect device IDs
    if device_ids is None:
        local_rank = get_local_rank()
        device_ids = [local_rank]

    if output_device is None:
        output_device = device_ids[0]

    # Wrap with DDP
    ddp_model = DDP(
        model,
        device_ids=device_ids,
        output_device=output_device,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=gradient_as_bucket_view,
        static_graph=static_graph,
    )

    print(
        f"DDP model initialized: device_ids={device_ids}, "
        f"output_device={output_device}"
    )

    return ddp_model


def save_ddp_checkpoint(
    model: DDP,
    optimizer: torch.optim.Optimizer,
    path: str,
    rank: int = 0,
) -> None:
    """
    Save DDP model checkpoint.

    Only rank 0 saves the checkpoint.

    Args:
        model: DDP model
        optimizer: Optimizer
        path: Path to save checkpoint
        rank: Current process rank
    """
    if rank == 0:
        # Extract underlying module from DDP wrapper
        model_state = model.module.state_dict()
        optimizer_state = optimizer.state_dict()

        checkpoint = {
            "model": model_state,
            "optimizer": optimizer_state,
        }

        torch.save(checkpoint, path)
        print(f"Saved DDP checkpoint to {path}")


def load_ddp_checkpoint(
    model: DDP,
    optimizer: torch.optim.Optimizer,
    path: str,
    map_location: Optional[str] = None,
) -> None:
    """
    Load DDP model checkpoint.

    Args:
        model: DDP model
        optimizer: Optimizer
        path: Path to load checkpoint from
        map_location: Device to map checkpoint to
    """
    # Load checkpoint
    checkpoint = torch.load(path, map_location=map_location)

    # Load into underlying module
    model.module.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    print(f"Loaded DDP checkpoint from {path}")


def get_model_from_ddp(model: nn.Module) -> nn.Module:
    """
    Extract the underlying model from DDP wrapper.

    Useful for saving/loading or accessing model-specific methods.

    Args:
        model: Potentially DDP-wrapped model

    Returns:
        Underlying model
    """
    if isinstance(model, DDP):
        return model.module
    return model
