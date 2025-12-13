"""
GRPO (Group Relative Policy Optimization) loss functions.

GRPO is a variant of PPO that normalizes rewards within groups to reduce variance.
This is particularly useful for LLM training where absolute reward magnitudes can vary.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def normalize_rewards_by_group(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Normalize rewards within each group to have zero mean and unit variance.

    Args:
        rewards: Tensor of shape (batch_size,) containing rewards
        group_ids: Tensor of shape (batch_size,) containing group IDs

    Returns:
        Normalized rewards tensor of shape (batch_size,)
    """
    normalized_rewards = torch.zeros_like(rewards)
    unique_groups = torch.unique(group_ids)

    for group_id in unique_groups:
        mask = group_ids == group_id
        group_rewards = rewards[mask]

        if len(group_rewards) > 1:
            # Normalize to zero mean and unit variance within group
            mean = group_rewards.mean()
            std = group_rewards.std() + 1e-8
            normalized_rewards[mask] = (group_rewards - mean) / std
        else:
            # Single sample in group - just center at zero
            normalized_rewards[mask] = 0.0

    return normalized_rewards


def compute_group_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    group_ids: torch.Tensor,
    gamma: float = 0.99,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute advantages using group-normalized rewards.

    For GRPO, we typically use simpler advantage estimation than GAE,
    focusing on group-relative comparisons.

    Args:
        rewards: Tensor of shape (batch_size,) containing rewards
        values: Tensor of shape (batch_size,) containing value estimates
        group_ids: Tensor of shape (batch_size,) containing group IDs
        gamma: Discount factor (often 1.0 for single-step GRPO)
        normalize: Whether to normalize advantages globally

    Returns:
        Advantages tensor of shape (batch_size,)
    """
    # Normalize rewards within groups
    normalized_rewards = normalize_rewards_by_group(rewards, group_ids)

    # Compute advantages: A = normalized_reward - V(s)
    advantages = normalized_rewards - values.detach()

    # Optionally normalize advantages globally
    if normalize:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages


def compute_grpo_loss(
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float = 0.2,
    group_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute the GRPO policy loss.

    GRPO uses the same clipped surrogate objective as PPO, but with
    group-normalized advantages to reduce variance.

    Args:
        current_log_probs: Log probabilities from current policy, shape (batch_size,)
        old_log_probs: Log probabilities from old policy, shape (batch_size,)
        advantages: Advantage estimates, shape (batch_size,)
        clip_range: PPO clipping range (epsilon)
        group_ids: Optional group IDs for additional group-wise statistics

    Returns:
        Tuple of (loss, info_dict)
    """
    # Compute probability ratio
    log_ratio = current_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    # Clipped surrogate objective
    surr1 = ratio * advantages
    ratio_clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    surr2 = ratio_clipped * advantages

    # Policy loss is negative because we want to maximize
    policy_loss = -torch.min(surr1, surr2).mean()

    # Compute statistics
    with torch.no_grad():
        # Fraction of samples where clipping was active
        clip_fraction = ((ratio < 1.0 - clip_range) | (ratio > 1.0 + clip_range)).float().mean()

        # KL divergence approximation
        approx_kl = ((ratio - 1) - log_ratio).mean()

        info = {
            "policy_loss": policy_loss.item(),
            "clip_fraction": clip_fraction.item(),
            "approx_kl": approx_kl.item(),
            "ratio_mean": ratio.mean().item(),
            "ratio_std": ratio.std().item(),
            "advantages_mean": advantages.mean().item(),
            "advantages_std": advantages.std().item(),
        }

        # Add group-wise statistics if group_ids provided
        if group_ids is not None:
            unique_groups = torch.unique(group_ids)
            group_advantages = []
            for group_id in unique_groups:
                mask = group_ids == group_id
                group_advantages.append(advantages[mask].mean().item())
            info["group_advantages_var"] = torch.tensor(group_advantages).var().item()

    return policy_loss, info


def compute_value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    clip_range: Optional[float] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute the value function loss.

    Args:
        values: Predicted values, shape (batch_size,)
        returns: Target returns, shape (batch_size,)
        clip_range: Optional value clipping range

    Returns:
        Tuple of (loss, info_dict)
    """
    if clip_range is not None:
        # Clipped value loss (similar to PPO)
        values_clipped = values + torch.clamp(
            values - returns, -clip_range, clip_range
        )
        loss1 = F.mse_loss(values, returns)
        loss2 = F.mse_loss(values_clipped, returns)
        value_loss = torch.max(loss1, loss2)
    else:
        # Standard MSE loss
        value_loss = F.mse_loss(values, returns)

    info = {
        "value_loss": value_loss.item(),
        "values_mean": values.mean().item(),
        "returns_mean": returns.mean().item(),
    }

    return value_loss, info


def compute_entropy_bonus(
    log_probs: torch.Tensor,
    entropy_coef: float = 0.01,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute entropy bonus for exploration.

    Args:
        log_probs: Log probabilities, shape (batch_size,)
        entropy_coef: Coefficient for entropy bonus

    Returns:
        Tuple of (entropy_loss, info_dict)

    Note:
        Returns negative entropy as loss (to be minimized).
    """
    # Approximate entropy from log probs
    # For language models, this is an approximation
    entropy = -log_probs.mean()
    entropy_loss = -entropy_coef * entropy  # Negative because we want to maximize entropy

    info = {
        "entropy": entropy.item(),
        "entropy_loss": entropy_loss.item(),
    }

    return entropy_loss, info
