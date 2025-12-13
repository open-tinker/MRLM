"""
FSDP (Fully Sharded Data Parallel) utilities.

FSDP shards model parameters, gradients, and optimizer states across GPUs,
enabling training of very large models that don't fit on a single GPU.
"""

from typing import Optional, Any, Type
from functools import partial

import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer


def get_transformer_layer_cls(model: nn.Module) -> Optional[Type[nn.Module]]:
    """
    Detect the transformer layer class for a model.

    Args:
        model: The model to inspect

    Returns:
        The transformer layer class, or None if not detected
    """
    # Map of model types to their transformer layer classes
    layer_cls_map = {
        "GPT2LMHeadModel": GPT2Block,
        "LlamaForCausalLM": LlamaDecoderLayer,
        "Qwen2ForCausalLM": Qwen2DecoderLayer,
    }

    model_type = model.__class__.__name__
    return layer_cls_map.get(model_type)


def get_fsdp_config(
    sharding_strategy: str = "full_shard",
    min_num_params: int = 1e8,
    transformer_layer_cls: Optional[Type[nn.Module]] = None,
    cpu_offload: bool = False,
    mixed_precision: bool = True,
) -> dict[str, Any]:
    """
    Get FSDP configuration.

    Args:
        sharding_strategy: Sharding strategy
            - "full_shard": Full sharding (FSDP)
            - "shard_grad_op": Shard gradients and optimizer states
            - "no_shard": DDP-like (no sharding)
        min_num_params: Minimum parameters for auto-wrapping
        transformer_layer_cls: Transformer layer class for wrapping
        cpu_offload: Whether to offload to CPU
        mixed_precision: Whether to use mixed precision

    Returns:
        Dictionary of FSDP configuration
    """
    from torch.distributed.fsdp import CPUOffload, MixedPrecision, BackwardPrefetch

    # Sharding strategy
    sharding_strategy_map = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
        "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
    }
    strategy = sharding_strategy_map.get(
        sharding_strategy, ShardingStrategy.FULL_SHARD
    )

    # Auto wrap policy
    if transformer_layer_cls is not None:
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={transformer_layer_cls},
        )
    else:
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params,
        )

    # CPU offload
    cpu_offload_config = CPUOffload(offload_params=True) if cpu_offload else None

    # Mixed precision
    mixed_precision_config = None
    if mixed_precision:
        mixed_precision_config = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )

    config = {
        "sharding_strategy": strategy,
        "auto_wrap_policy": auto_wrap_policy,
        "cpu_offload": cpu_offload_config,
        "mixed_precision": mixed_precision_config,
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "device_id": torch.cuda.current_device(),
        "limit_all_gathers": True,
    }

    return config


def setup_fsdp_model(
    model: nn.Module,
    sharding_strategy: str = "full_shard",
    cpu_offload: bool = False,
    mixed_precision: bool = True,
    auto_detect_layer: bool = True,
) -> FSDP:
    """
    Wrap model with FSDP.

    Args:
        model: Model to wrap
        sharding_strategy: Sharding strategy (see get_fsdp_config)
        cpu_offload: Whether to offload parameters to CPU
        mixed_precision: Whether to use mixed precision training
        auto_detect_layer: Whether to auto-detect transformer layer class

    Returns:
        FSDP-wrapped model

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> fsdp_model = setup_fsdp_model(model)
    """
    # Auto-detect transformer layer class
    transformer_layer_cls = None
    if auto_detect_layer:
        transformer_layer_cls = get_transformer_layer_cls(model)
        if transformer_layer_cls is not None:
            print(f"Auto-detected transformer layer: {transformer_layer_cls.__name__}")

    # Get FSDP config
    fsdp_config = get_fsdp_config(
        sharding_strategy=sharding_strategy,
        transformer_layer_cls=transformer_layer_cls,
        cpu_offload=cpu_offload,
        mixed_precision=mixed_precision,
    )

    # Wrap with FSDP
    fsdp_model = FSDP(model, **fsdp_config)

    print(
        f"FSDP model initialized: strategy={sharding_strategy}, "
        f"cpu_offload={cpu_offload}, mixed_precision={mixed_precision}"
    )

    return fsdp_model


def save_fsdp_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    path: str,
    rank: int = 0,
) -> None:
    """
    Save FSDP model checkpoint.

    Only the main process (rank 0) saves the full checkpoint.

    Args:
        model: FSDP model
        optimizer: Optimizer
        path: Path to save checkpoint
        rank: Current process rank
    """
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    # Configure to save full state dict on rank 0
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        model_state = model.state_dict()
        optimizer_state = optimizer.state_dict()

        if rank == 0:
            checkpoint = {
                "model": model_state,
                "optimizer": optimizer_state,
            }
            torch.save(checkpoint, path)
            print(f"Saved FSDP checkpoint to {path}")


def load_fsdp_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    path: str,
) -> None:
    """
    Load FSDP model checkpoint.

    Args:
        model: FSDP model
        optimizer: Optimizer
        path: Path to load checkpoint from
    """
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType

    # Load checkpoint
    checkpoint = torch.load(path, map_location="cpu")

    # Configure to load full state dict
    load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, load_policy):
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    print(f"Loaded FSDP checkpoint from {path}")
