"""
PPO loss functions.

This module implements the loss functions for Proximal Policy Optimization (PPO),
including the clipped surrogate objective, value loss, and entropy bonus.
"""

import torch
import torch.nn.functional as F


def compute_ppo_loss(
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float = 0.2,
) -> torch.Tensor:
    """
    Compute PPO clipped surrogate objective.

    The PPO objective limits how much the policy can change in a single update,
    preventing destructively large policy updates.

    L^CLIP(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]

    where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)

    Args:
        current_log_probs: Log probabilities under current policy
        old_log_probs: Log probabilities under old policy
        advantages: Advantage estimates
        clip_range: Clipping parameter ε (typically 0.1-0.3)

    Returns:
        PPO policy loss (negative, for minimization)

    References:
        Schulman et al., "Proximal Policy Optimization Algorithms", 2017
    """
    # Compute probability ratio: π_new / π_old
    log_ratio = current_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    # Unclipped objective
    surr1 = ratio * advantages

    # Clipped objective
    ratio_clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    surr2 = ratio_clipped * advantages

    # Take minimum (pessimistic bound)
    policy_loss = -torch.min(surr1, surr2).mean()

    return policy_loss


def compute_value_loss(
    predicted_values: torch.Tensor,
    returns: torch.Tensor,
    clip_range: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute value function loss.

    Standard MSE loss between predicted values and returns.
    Optionally clips value function updates for stability.

    Args:
        predicted_values: Value estimates from critic
        returns: Computed returns (targets)
        clip_range: Optional clipping range for value function

    Returns:
        Value function loss
    """
    if clip_range is not None:
        # Clipped value loss (similar to policy clipping)
        # Prevents large value function updates
        value_loss_unclipped = (predicted_values - returns) ** 2
        value_clipped = predicted_values + torch.clamp(
            predicted_values - predicted_values,
            -clip_range,
            clip_range,
        )
        value_loss_clipped = (value_clipped - returns) ** 2
        value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
    else:
        # Standard MSE loss
        value_loss = F.mse_loss(predicted_values, returns)

    return value_loss


def compute_entropy_bonus(log_probs: torch.Tensor, probs: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute entropy bonus for exploration.

    Entropy H = -Σ p(a) * log p(a)

    Higher entropy = more exploration (less deterministic policy)

    Args:
        log_probs: Log probabilities of actions
        probs: Probabilities of actions (if available)

    Returns:
        Negative entropy (for minimization, so we subtract it from loss)
    """
    if probs is not None:
        # Use probabilities directly for more stable computation
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
    else:
        # Approximate from log probs
        # H ≈ -mean(log_probs)
        entropy = -log_probs.mean()

    # Return negative (we want to maximize entropy, so minimize -entropy)
    return -entropy


def compute_ppo_total_loss(
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    predicted_values: torch.Tensor,
    returns: torch.Tensor,
    clip_range: float = 0.2,
    value_loss_coef: float = 0.5,
    entropy_coef: float = 0.01,
    value_clip_range: Optional[float] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute total PPO loss combining all components.

    Total loss = Policy loss + value_coef * Value loss - entropy_coef * Entropy

    Args:
        current_log_probs: Log probs under current policy
        old_log_probs: Log probs under old policy
        advantages: Advantage estimates
        predicted_values: Value predictions
        returns: Target returns
        clip_range: Policy clipping parameter
        value_loss_coef: Coefficient for value loss
        entropy_coef: Coefficient for entropy bonus
        value_clip_range: Optional value function clipping

    Returns:
        Tuple of (total_loss, metrics_dict)

    Example:
        >>> loss, metrics = compute_ppo_total_loss(
        ...     current_log_probs, old_log_probs, advantages,
        ...     values, returns
        ... )
        >>> loss.backward()
    """
    # Compute individual losses
    policy_loss = compute_ppo_loss(current_log_probs, old_log_probs, advantages, clip_range)

    value_loss = compute_value_loss(predicted_values, returns, value_clip_range)

    entropy = compute_entropy_bonus(current_log_probs)

    # Combine losses
    total_loss = policy_loss + value_loss_coef * value_loss + entropy_coef * entropy

    # Collect metrics
    metrics = {
        "loss/policy": policy_loss.item(),
        "loss/value": value_loss.item(),
        "loss/entropy": -entropy.item(),  # Report actual entropy (positive)
        "loss/total": total_loss.item(),
    }

    # Additional diagnostics
    with torch.no_grad():
        # Approximate KL divergence: KL ≈ (log_ratio)^2 / 2
        log_ratio = current_log_probs - old_log_probs
        approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
        metrics["diagnostics/approx_kl"] = approx_kl.item()

        # Clip fraction (how often is clipping active?)
        ratio = torch.exp(log_ratio)
        clip_fraction = (
            (torch.abs(ratio - 1.0) > clip_range).float().mean()
        )
        metrics["diagnostics/clip_fraction"] = clip_fraction.item()

        # Explained variance
        explained_var = 1 - (returns - predicted_values).var() / (returns.var() + 1e-8)
        metrics["diagnostics/explained_variance"] = explained_var.item()

    return total_loss, metrics


# Type hints
from typing import Optional
