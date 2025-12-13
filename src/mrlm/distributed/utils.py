"""
Distributed training utilities.

Helper functions for common distributed operations.
"""

from typing import Optional, List, Any

import torch
import torch.distributed as dist

from mrlm.distributed.setup import is_distributed, get_rank, get_world_size


def all_reduce_mean(tensor: torch.Tensor, world_size: Optional[int] = None) -> torch.Tensor:
    """
    All-reduce a tensor and compute the mean across all processes.

    Args:
        tensor: Tensor to reduce
        world_size: Number of processes (auto-detected if None)

    Returns:
        Averaged tensor
    """
    if not is_distributed():
        return tensor

    if world_size is None:
        world_size = get_world_size()

    # Clone to avoid modifying input
    tensor = tensor.clone()

    # All-reduce
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # Average
    tensor = tensor / world_size

    return tensor


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-reduce a tensor and compute the sum across all processes.

    Args:
        tensor: Tensor to reduce

    Returns:
        Summed tensor
    """
    if not is_distributed():
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def barrier() -> None:
    """
    Synchronization barrier - wait for all processes.

    Useful for ensuring all processes reach a point before continuing.
    """
    if is_distributed():
        dist.barrier()


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """
    Broadcast tensor from source process to all processes.

    Args:
        tensor: Tensor to broadcast
        src: Source rank

    Returns:
        Broadcasted tensor
    """
    if not is_distributed():
        return tensor

    dist.broadcast(tensor, src=src)
    return tensor


def gather_tensors(tensor: torch.Tensor, dst: int = 0) -> Optional[List[torch.Tensor]]:
    """
    Gather tensors from all processes to destination process.

    Args:
        tensor: Tensor to gather
        dst: Destination rank

    Returns:
        List of tensors from all processes (only on dst rank, None otherwise)
    """
    if not is_distributed():
        return [tensor]

    world_size = get_world_size()
    rank = get_rank()

    # Prepare gather list on destination
    if rank == dst:
        gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    else:
        gather_list = None

    # Gather
    dist.gather(tensor, gather_list, dst=dst)

    return gather_list


def all_gather_tensors(tensor: torch.Tensor) -> List[torch.Tensor]:
    """
    Gather tensors from all processes to all processes.

    Args:
        tensor: Tensor to gather

    Returns:
        List of tensors from all processes
    """
    if not is_distributed():
        return [tensor]

    world_size = get_world_size()
    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]

    dist.all_gather(gather_list, tensor)

    return gather_list


def reduce_dict(input_dict: dict[str, torch.Tensor], average: bool = True) -> dict[str, torch.Tensor]:
    """
    Reduce a dictionary of tensors across all processes.

    Args:
        input_dict: Dictionary of tensors to reduce
        average: Whether to average (True) or sum (False)

    Returns:
        Reduced dictionary
    """
    if not is_distributed():
        return input_dict

    world_size = get_world_size()
    output_dict = {}

    for key, value in input_dict.items():
        # Convert to tensor if needed
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.float32)

        # Reduce
        reduced = all_reduce_sum(value)

        # Average if requested
        if average:
            reduced = reduced / world_size

        output_dict[key] = reduced

    return output_dict


class GradientAccumulator:
    """
    Utility for gradient accumulation in distributed training.

    Gradient accumulation allows simulating larger batch sizes by
    accumulating gradients over multiple micro-batches.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
    ):
        """
        Initialize gradient accumulator.

        Args:
            model: Model being trained
            optimizer: Optimizer
            accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.step_count = 0

    def step(self, loss: torch.Tensor) -> bool:
        """
        Perform a training step with gradient accumulation.

        Args:
            loss: Loss value

        Returns:
            True if optimizer step was performed (accumulation complete)
        """
        # Scale loss by accumulation steps
        scaled_loss = loss / self.accumulation_steps

        # Backward pass
        scaled_loss.backward()

        # Increment counter
        self.step_count += 1

        # Check if we should update weights
        if self.step_count % self.accumulation_steps == 0:
            # Clip gradients
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

            return True

        return False

    def reset(self) -> None:
        """Reset gradient accumulation state."""
        self.step_count = 0
        self.optimizer.zero_grad()


def setup_for_distributed_training(
    model: torch.nn.Module,
    strategy: str = "ddp",
    **kwargs,
) -> torch.nn.Module:
    """
    Convenience function to setup model for distributed training.

    Args:
        model: Model to setup
        strategy: Distribution strategy ('ddp', 'fsdp', 'none')
        **kwargs: Additional arguments for setup functions

    Returns:
        Distributed model

    Example:
        >>> model = setup_for_distributed_training(model, strategy="fsdp")
    """
    from mrlm.distributed.ddp import setup_ddp_model
    from mrlm.distributed.fsdp import setup_fsdp_model

    if strategy == "ddp":
        return setup_ddp_model(model, **kwargs)
    elif strategy == "fsdp":
        return setup_fsdp_model(model, **kwargs)
    elif strategy == "none":
        return model
    else:
        raise ValueError(f"Unknown distribution strategy: {strategy}")
