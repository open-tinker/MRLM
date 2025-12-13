"""
SFT (Supervised Fine-Tuning) loss functions.

Supports:
- Behavioral cloning: Train model to predict actions given observations
- Next state prediction: Train model to predict next observations (world model)
- Standard language modeling loss
"""

from typing import Tuple, Optional, Dict

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from mrlm.core.types import Message


def compute_sft_loss(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute standard supervised fine-tuning loss (next token prediction).

    Args:
        model: Language model
        input_ids: Input token IDs, shape (batch_size, seq_len)
        attention_mask: Attention mask, shape (batch_size, seq_len)
        labels: Target token IDs, shape (batch_size, seq_len)
                If None, uses shifted input_ids

    Returns:
        Tuple of (loss, info_dict)
    """
    # Default labels to shifted inputs
    if labels is None:
        labels = input_ids.clone()

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    loss = outputs.loss

    # Compute metrics
    with torch.no_grad():
        # Perplexity
        perplexity = torch.exp(loss)

        info = {
            "loss": loss.item(),
            "perplexity": perplexity.item(),
        }

    return loss, info


def compute_behavioral_cloning_loss(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    observations: list[Message],
    actions: list[Message],
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute behavioral cloning loss.

    Train model to predict action given observation (state -> action).

    Args:
        model: Language model
        tokenizer: Tokenizer
        observations: List of observation messages (prompts)
        actions: List of action messages (target responses)
        device: Device for computation

    Returns:
        Tuple of (loss, info_dict)
    """
    from mrlm.models.generation import format_messages_for_chat

    batch_losses = []

    for obs, action in zip(observations, actions):
        # Format observation as prompt
        prompt = format_messages_for_chat([obs], tokenizer)

        # Create full sequence: prompt + action
        full_text = prompt + action.content

        # Tokenize
        prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)["input_ids"]
        full_ids = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)["input_ids"]

        # Move to device
        prompt_ids = prompt_ids.to(device)
        full_ids = full_ids.to(device)

        # Create labels: -100 for prompt tokens (ignore), actual IDs for action tokens
        labels = full_ids.clone()
        prompt_length = prompt_ids.shape[1]
        labels[:, :prompt_length] = -100  # Ignore prompt in loss

        # Forward pass
        outputs = model(input_ids=full_ids, labels=labels)
        batch_losses.append(outputs.loss)

    # Average loss
    loss = torch.stack(batch_losses).mean()

    # Metrics
    with torch.no_grad():
        info = {
            "bc_loss": loss.item(),
            "perplexity": torch.exp(loss).item(),
        }

    return loss, info


def compute_next_state_loss(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    observations: list[Message],
    actions: list[Message],
    next_observations: list[Message],
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute next state prediction loss (world model).

    Train model to predict next observation given current observation and action.
    This is useful for world model training.

    Args:
        model: Language model
        tokenizer: Tokenizer
        observations: List of current observation messages
        actions: List of action messages
        next_observations: List of next observation messages (targets)
        device: Device for computation

    Returns:
        Tuple of (loss, info_dict)
    """
    from mrlm.models.generation import format_messages_for_chat

    batch_losses = []

    for obs, action, next_obs in zip(observations, actions, next_observations):
        # Format context: observation + action
        context_messages = [obs, action]
        context = format_messages_for_chat(context_messages, tokenizer)

        # Extract next observation content (what we want to predict)
        if next_obs.messages:
            # Get the latest message from next observation
            target_text = next_obs.messages[-1].content
        else:
            # Skip if no messages in next observation
            continue

        # Create full sequence
        full_text = context + target_text

        # Tokenize
        context_ids = tokenizer(context, return_tensors="pt", truncation=True, max_length=2048)["input_ids"]
        full_ids = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)["input_ids"]

        # Move to device
        context_ids = context_ids.to(device)
        full_ids = full_ids.to(device)

        # Create labels: ignore context, predict next state
        labels = full_ids.clone()
        context_length = context_ids.shape[1]
        labels[:, :context_length] = -100

        # Forward pass
        outputs = model(input_ids=full_ids, labels=labels)
        batch_losses.append(outputs.loss)

    if not batch_losses:
        # No valid samples
        return torch.tensor(0.0, device=device), {"next_state_loss": 0.0}

    # Average loss
    loss = torch.stack(batch_losses).mean()

    # Metrics
    with torch.no_grad():
        info = {
            "next_state_loss": loss.item(),
            "perplexity": torch.exp(loss).item(),
        }

    return loss, info


def compute_combined_sft_loss(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    observations: list[Message],
    actions: list[Message],
    next_observations: list[Message],
    device: torch.device,
    bc_weight: float = 0.5,
    world_model_weight: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute combined behavioral cloning + world model loss.

    Args:
        model: Language model
        tokenizer: Tokenizer
        observations: Current observations
        actions: Actions taken
        next_observations: Next observations
        device: Device
        bc_weight: Weight for behavioral cloning loss
        world_model_weight: Weight for world model loss

    Returns:
        Tuple of (total_loss, info_dict)
    """
    # Behavioral cloning loss
    bc_loss, bc_info = compute_behavioral_cloning_loss(
        model=model,
        tokenizer=tokenizer,
        observations=observations,
        actions=actions,
        device=device,
    )

    # World model loss
    wm_loss, wm_info = compute_next_state_loss(
        model=model,
        tokenizer=tokenizer,
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        device=device,
    )

    # Combined loss
    total_loss = bc_weight * bc_loss + world_model_weight * wm_loss

    # Combined info
    info = {
        "total_loss": total_loss.item(),
        "bc_loss": bc_info["bc_loss"],
        "world_model_loss": wm_info["next_state_loss"],
        "bc_weight": bc_weight,
        "wm_weight": world_model_weight,
    }

    return total_loss, info
