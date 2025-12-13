"""GPU selection and management utilities."""

import os
import torch
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def get_available_gpus() -> List[int]:
    """Get list of available GPU IDs.

    Returns:
        List of available GPU device IDs

    Example:
        >>> gpus = get_available_gpus()
        >>> print(f"Available GPUs: {gpus}")
        Available GPUs: [0, 1, 2, 3]
    """
    if not torch.cuda.is_available():
        return []

    return list(range(torch.cuda.device_count()))


def get_gpu_info() -> List[dict]:
    """Get detailed information about all available GPUs.

    Returns:
        List of dictionaries containing GPU information

    Example:
        >>> info = get_gpu_info()
        >>> for gpu in info:
        ...     print(f"GPU {gpu['id']}: {gpu['name']} ({gpu['memory_total']} MB)")
    """
    if not torch.cuda.is_available():
        return []

    gpu_info = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        info = {
            "id": i,
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "memory_total": props.total_memory // (1024**2),  # MB
            "multi_processor_count": props.multi_processor_count,
        }
        gpu_info.append(info)

    return gpu_info


def select_gpus(
    gpu_ids: Optional[List[int]] = None,
    num_gpus: Optional[int] = None,
    min_memory_mb: Optional[int] = None,
) -> List[int]:
    """Select GPUs based on various criteria.

    Args:
        gpu_ids: Specific GPU IDs to use. If None, auto-select.
        num_gpus: Number of GPUs to use. If None, use all available.
        min_memory_mb: Minimum GPU memory required (MB)

    Returns:
        List of selected GPU IDs

    Raises:
        ValueError: If requested GPUs are not available

    Example:
        >>> # Use specific GPUs
        >>> gpus = select_gpus(gpu_ids=[0, 2])

        >>> # Use first 2 GPUs
        >>> gpus = select_gpus(num_gpus=2)

        >>> # Use GPUs with at least 16GB memory
        >>> gpus = select_gpus(min_memory_mb=16000)
    """
    available = get_available_gpus()

    if not available:
        logger.warning("No GPUs available, will use CPU")
        return []

    # If specific GPU IDs requested
    if gpu_ids is not None:
        # Validate requested GPUs
        invalid = [gid for gid in gpu_ids if gid not in available]
        if invalid:
            raise ValueError(
                f"Requested GPU IDs {invalid} are not available. "
                f"Available GPUs: {available}"
            )
        selected = gpu_ids
    else:
        # Auto-select based on criteria
        selected = available.copy()

        # Filter by memory if specified
        if min_memory_mb is not None:
            gpu_info = get_gpu_info()
            selected = [
                info["id"]
                for info in gpu_info
                if info["memory_total"] >= min_memory_mb
            ]

            if not selected:
                raise ValueError(
                    f"No GPUs with at least {min_memory_mb}MB memory found"
                )

        # Limit number of GPUs if specified
        if num_gpus is not None:
            if num_gpus > len(selected):
                logger.warning(
                    f"Requested {num_gpus} GPUs but only {len(selected)} available"
                )
            selected = selected[:num_gpus]

    return selected


def set_visible_gpus(gpu_ids: List[int]) -> None:
    """Set visible GPUs using CUDA_VISIBLE_DEVICES.

    This restricts PyTorch to only see the specified GPUs.
    Must be called before any CUDA operations.

    Args:
        gpu_ids: List of GPU IDs to make visible

    Example:
        >>> set_visible_gpus([0, 2, 3])  # Only GPUs 0, 2, 3 visible
        >>> # Now torch.cuda.device_count() will return 3
    """
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        logger.info(f"Set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        # No GPUs - use CPU only
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logger.info("Set CUDA_VISIBLE_DEVICES='' (CPU only)")


def get_optimal_device(gpu_ids: Optional[List[int]] = None) -> torch.device:
    """Get optimal device for training.

    Args:
        gpu_ids: Specific GPU IDs to consider. If None, use all available.

    Returns:
        torch.device object (cuda:X or cpu)

    Example:
        >>> device = get_optimal_device()
        >>> model.to(device)
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")

    if gpu_ids is None:
        # Use first GPU by default
        return torch.device("cuda:0")

    if not gpu_ids:
        return torch.device("cpu")

    # Use first GPU from list
    return torch.device(f"cuda:{gpu_ids[0]}")


def get_gpu_memory_usage(device_id: int = 0) -> Tuple[int, int, float]:
    """Get current GPU memory usage.

    Args:
        device_id: GPU device ID

    Returns:
        Tuple of (allocated_mb, total_mb, utilization_percent)

    Example:
        >>> allocated, total, util = get_gpu_memory_usage(0)
        >>> print(f"GPU 0: {allocated}/{total} MB ({util:.1f}% used)")
    """
    if not torch.cuda.is_available():
        return 0, 0, 0.0

    allocated = torch.cuda.memory_allocated(device_id) // (1024**2)
    total = torch.cuda.get_device_properties(device_id).total_memory // (1024**2)
    utilization = (allocated / total * 100) if total > 0 else 0.0

    return allocated, total, utilization


def print_gpu_info(gpu_ids: Optional[List[int]] = None) -> None:
    """Print information about available GPUs.

    Args:
        gpu_ids: Specific GPU IDs to show. If None, show all.

    Example:
        >>> print_gpu_info()
        Available GPUs: 4
        GPU 0: NVIDIA A100-SXM4-40GB (40536 MB, Compute 8.0)
        GPU 1: NVIDIA A100-SXM4-40GB (40536 MB, Compute 8.0)
        ...
    """
    if not torch.cuda.is_available():
        print("No CUDA GPUs available")
        return

    gpu_info = get_gpu_info()

    if gpu_ids is not None:
        gpu_info = [info for info in gpu_info if info["id"] in gpu_ids]

    print(f"Available GPUs: {len(gpu_info)}")
    for info in gpu_info:
        allocated, total, util = get_gpu_memory_usage(info["id"])
        print(
            f"GPU {info['id']}: {info['name']} "
            f"({info['memory_total']} MB total, {allocated} MB used ({util:.1f}%), "
            f"Compute {info['compute_capability']})"
        )


def auto_select_gpus_for_model(
    model_size_gb: float,
    safety_margin: float = 1.5,
) -> List[int]:
    """Automatically select GPUs based on model size.

    Args:
        model_size_gb: Estimated model size in GB
        safety_margin: Multiply model size by this for safety (default: 1.5x)

    Returns:
        List of GPU IDs that can fit the model

    Example:
        >>> # For a 7B parameter model (~14GB)
        >>> gpus = auto_select_gpus_for_model(model_size_gb=14)
    """
    required_mb = int(model_size_gb * 1024 * safety_margin)

    try:
        return select_gpus(min_memory_mb=required_mb)
    except ValueError:
        # If no single GPU has enough memory, return all GPUs for FSDP
        logger.warning(
            f"No single GPU has {required_mb}MB. "
            "Returning all GPUs for FSDP/model parallelism."
        )
        return get_available_gpus()


def setup_gpu_environment(
    gpu_ids: Optional[List[int]] = None,
    num_gpus: Optional[int] = None,
) -> torch.device:
    """Setup GPU environment for training.

    This is a convenience function that:
    1. Selects GPUs based on criteria
    2. Sets CUDA_VISIBLE_DEVICES
    3. Returns optimal device

    Args:
        gpu_ids: Specific GPU IDs to use
        num_gpus: Number of GPUs to use

    Returns:
        torch.device for model placement

    Example:
        >>> device = setup_gpu_environment(num_gpus=2)
        >>> model.to(device)
    """
    selected_gpus = select_gpus(gpu_ids=gpu_ids, num_gpus=num_gpus)

    if selected_gpus:
        set_visible_gpus(selected_gpus)
        print_gpu_info(selected_gpus)
    else:
        logger.info("No GPUs selected, using CPU")

    return get_optimal_device(selected_gpus)
