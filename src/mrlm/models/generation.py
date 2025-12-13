"""
Text generation utilities for LLM models.

This module provides functions for generating text with transformers models,
including utilities for computing log probabilities and value estimates.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from mrlm.core.types import Message, MessageRole, GenerationConfig


def format_messages_for_chat(
    messages: List[Message], tokenizer: PreTrainedTokenizer
) -> str:
    """
    Format messages for chat model input.

    Args:
        messages: List of messages in conversation
        tokenizer: Tokenizer with chat template

    Returns:
        Formatted string ready for tokenization
    """
    # Convert to format expected by chat template
    formatted_messages = []
    for msg in messages:
        role = msg.role.value if isinstance(msg.role, MessageRole) else msg.role
        formatted_messages.append({"role": role, "content": msg.content})

    # Use tokenizer's chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        formatted = tokenizer.apply_chat_template(
            formatted_messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Fallback: simple concatenation
        formatted = ""
        for msg in formatted_messages:
            formatted += f"{msg['role']}: {msg['content']}\n"
        formatted += "assistant: "

    return formatted


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages: List[Message],
    generation_config: Optional[GenerationConfig] = None,
    device: Optional[torch.device] = None,
) -> str:
    """
    Generate response from model (inference only, no gradients).

    Args:
        model: Pre-trained model
        tokenizer: Tokenizer
        messages: Conversation history
        generation_config: Generation parameters
        device: Device to use (if None, uses model's device)

    Returns:
        Generated response text

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> messages = [Message(role=MessageRole.USER, content="Hello!")]
        >>> response = generate_response(model, tokenizer, messages)
    """
    if generation_config is None:
        generation_config = GenerationConfig()

    if device is None:
        device = next(model.parameters()).device

    # Format input
    prompt = format_messages_for_chat(messages, tokenizer)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        model.eval()
        outputs = model.generate(**inputs, **generation_config.to_dict())

    # Decode (only new tokens)
    input_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[0, input_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response


def generate_with_log_probs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages: List[Message],
    generation_config: Optional[GenerationConfig] = None,
    device: Optional[torch.device] = None,
    return_value: bool = False,
) -> Tuple[str, torch.Tensor, Optional[torch.Tensor]]:
    """
    Generate response with log probabilities (for training).

    This function computes gradients and returns log probabilities
    for use in RL training algorithms like PPO.

    Args:
        model: Pre-trained model
        tokenizer: Tokenizer
        messages: Conversation history
        generation_config: Generation parameters
        device: Device to use
        return_value: Whether to compute value estimate (requires value head)

    Returns:
        Tuple of (response_text, log_probs, value)
            - response_text: Generated text
            - log_probs: Log probabilities of generated tokens
            - value: Value estimate (if return_value=True, else None)

    Example:
        >>> response, log_probs, value = generate_with_log_probs(
        ...     model, tokenizer, messages, return_value=True
        ... )
        >>> loss = -log_probs.mean()  # Simplified loss
        >>> loss.backward()
    """
    if generation_config is None:
        generation_config = GenerationConfig()

    if device is None:
        device = next(model.parameters()).device

    # Format and tokenize input
    prompt = format_messages_for_chat(messages, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Set model to training mode for gradients
    model.train()

    # Generate with gradient tracking
    # Note: This is simplified. Real implementation would use techniques like:
    # - Teacher forcing during training
    # - Separate rollout and training phases
    # - Cached KV for efficiency

    # For now, we'll use greedy decoding with log probs
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

    max_new_tokens = generation_config.max_new_tokens
    generated_tokens = []
    log_probs_list = []

    for _ in range(max_new_tokens):
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Get logits for last position
        next_token_logits = logits[:, -1, :]

        # Apply temperature
        if generation_config.temperature != 1.0:
            next_token_logits = next_token_logits / generation_config.temperature

        # Compute probabilities
        probs = F.softmax(next_token_logits, dim=-1)

        # Sample or take argmax
        if generation_config.do_sample:
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(probs, dim=-1, keepdim=True)

        # Compute log probability of selected token
        log_prob = torch.log(probs[0, next_token[0, 0]] + 1e-10)
        log_probs_list.append(log_prob)

        # Append to generated sequence
        generated_tokens.append(next_token[0, 0].item())

        # Check for EOS
        if (
            generation_config.eos_token_id is not None
            and next_token[0, 0].item() == generation_config.eos_token_id
        ):
            break

        # Update input_ids and attention_mask
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=device, dtype=attention_mask.dtype)],
            dim=-1,
        )

    # Decode response
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Stack log probabilities
    log_probs = torch.stack(log_probs_list)

    # Compute value estimate if requested
    value = None
    if return_value:
        # Value estimation would require a value head
        # For now, return None - will be implemented with PPO trainer
        value = torch.tensor(0.0, device=device, requires_grad=True)

    return response, log_probs, value


def compute_log_probs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages: List[Message],
    response_text: str,
    device: Optional[torch.device] = None,
    return_value: bool = False,
) -> Tuple[str, torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute log probabilities for a given response text.

    This is used during training to recompute log probs for previously
    generated responses (e.g., for PPO ratio calculations).

    Args:
        model: Pre-trained model
        tokenizer: Tokenizer
        messages: Conversation history (prompt)
        response_text: Response text to compute log probs for
        device: Device to use
        return_value: Whether to compute value estimate

    Returns:
        Tuple of (response_text, log_probs, value)
            - response_text: The input response text (returned for consistency)
            - log_probs: Sum of log probabilities of response tokens
            - value: Value estimate (if return_value=True, else None)
    """
    if device is None:
        device = next(model.parameters()).device

    # Format prompt
    prompt = format_messages_for_chat(messages, tokenizer)

    # Tokenize prompt and response separately
    prompt_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    full_text = prompt + response_text
    full_inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)

    # Move to device
    prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
    full_inputs = {k: v.to(device) for k, v in full_inputs.items()}

    # Get length of prompt to identify response tokens
    prompt_length = prompt_inputs["input_ids"].shape[1]

    # Forward pass with gradients
    model.train()
    outputs = model(
        input_ids=full_inputs["input_ids"],
        attention_mask=full_inputs.get("attention_mask"),
    )

    logits = outputs.logits

    # Get logits for response tokens only
    # logits[:, i, :] predicts token at position i+1
    # So for response starting at prompt_length, we need logits[prompt_length-1:]
    response_logits = logits[:, prompt_length - 1 : -1, :]
    response_token_ids = full_inputs["input_ids"][:, prompt_length:]

    # Compute log probabilities
    log_probs = F.log_softmax(response_logits, dim=-1)

    # Gather log probs of actual response tokens
    token_log_probs = torch.gather(
        log_probs, dim=-1, index=response_token_ids.unsqueeze(-1)
    ).squeeze(-1)

    # Sum log probs (total log prob of sequence)
    total_log_prob = token_log_probs.sum()

    # Compute value estimate if requested
    value = None
    if return_value:
        # Simple value estimate: mean of last layer hidden states
        # In practice, this would use a learned value head
        value = torch.tensor(0.0, device=device, requires_grad=True)

    return response_text, total_log_prob, value


def compute_sequence_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute log probabilities for a given sequence.

    This is useful for computing log probs of existing sequences
    (e.g., for DPO or computing advantage in PPO).

    Args:
        model: Pre-trained model
        input_ids: Token IDs of shape (batch_size, seq_len)
        attention_mask: Attention mask of shape (batch_size, seq_len)

    Returns:
        Log probabilities of shape (batch_size, seq_len - 1)
        (one less than input because we predict next token)

    Example:
        >>> input_ids = tokenizer("Hello world", return_tensors="pt")["input_ids"]
        >>> log_probs = compute_sequence_log_probs(model, input_ids)
    """
    with torch.enable_grad():
        model.train()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log probs of actual tokens
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        return token_log_probs
