"""
DPO (Direct Preference Optimization) loss functions.

DPO directly optimizes for preferences without requiring online RL rollouts.
The key insight is to optimize the policy to maximize the likelihood of
preferred responses over non-preferred ones, while staying close to a reference model.
"""

from typing import Tuple, Optional

import torch
import torch.nn.functional as F


def compute_preference_log_ratio(
    chosen_log_probs: torch.Tensor,
    rejected_log_probs: torch.Tensor,
    chosen_ref_log_probs: torch.Tensor,
    rejected_ref_log_probs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute log probability ratios for DPO.

    Args:
        chosen_log_probs: Log probs from policy for chosen responses
        rejected_log_probs: Log probs from policy for rejected responses
        chosen_ref_log_probs: Log probs from reference model for chosen responses
        rejected_ref_log_probs: Log probs from reference model for rejected responses

    Returns:
        Tuple of (chosen_log_ratio, rejected_log_ratio)
            - chosen_log_ratio: log(π_θ(y_w|x) / π_ref(y_w|x))
            - rejected_log_ratio: log(π_θ(y_l|x) / π_ref(y_l|x))
    """
    chosen_log_ratio = chosen_log_probs - chosen_ref_log_probs
    rejected_log_ratio = rejected_log_probs - rejected_ref_log_probs

    return chosen_log_ratio, rejected_log_ratio


def compute_dpo_loss(
    chosen_log_probs: torch.Tensor,
    rejected_log_probs: torch.Tensor,
    chosen_ref_log_probs: torch.Tensor,
    rejected_ref_log_probs: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute the DPO loss.

    DPO loss: -log(σ(β * [log(π_θ(y_w|x)/π_ref(y_w|x)) - log(π_θ(y_l|x)/π_ref(y_l|x))]))

    where:
    - y_w is the winning/chosen response
    - y_l is the losing/rejected response
    - π_θ is the current policy
    - π_ref is the reference policy
    - β is the temperature parameter (controls KL penalty strength)
    - σ is the sigmoid function

    Args:
        chosen_log_probs: Log probs from policy for chosen responses, shape (batch_size,)
        rejected_log_probs: Log probs from policy for rejected responses, shape (batch_size,)
        chosen_ref_log_probs: Log probs from reference for chosen responses, shape (batch_size,)
        rejected_ref_log_probs: Log probs from reference for rejected responses, shape (batch_size,)
        beta: Temperature parameter (higher = stronger KL penalty)
        label_smoothing: Label smoothing parameter (0 = no smoothing)

    Returns:
        Tuple of (loss, info_dict)
    """
    # Compute log ratios
    chosen_log_ratio, rejected_log_ratio = compute_preference_log_ratio(
        chosen_log_probs=chosen_log_probs,
        rejected_log_probs=rejected_log_probs,
        chosen_ref_log_probs=chosen_ref_log_probs,
        rejected_ref_log_probs=rejected_ref_log_probs,
    )

    # Compute logits for preference model
    # logit = β * (log(π/π_ref)[chosen] - log(π/π_ref)[rejected])
    logits = beta * (chosen_log_ratio - rejected_log_ratio)

    # DPO loss: -log(sigmoid(logits))
    # With label smoothing: -(1-ε)log(sigmoid(logits)) - ε*log(1-sigmoid(logits))
    if label_smoothing > 0:
        # Binary cross entropy with label smoothing
        loss = (
            -(1 - label_smoothing) * F.logsigmoid(logits)
            - label_smoothing * F.logsigmoid(-logits)
        ).mean()
    else:
        # Standard DPO loss
        loss = -F.logsigmoid(logits).mean()

    # Compute statistics
    with torch.no_grad():
        # Accuracy: how often does model prefer chosen over rejected?
        accuracy = (logits > 0).float().mean()

        # Reward margins (implicit rewards)
        chosen_rewards = beta * chosen_log_ratio
        rejected_rewards = beta * rejected_log_ratio
        reward_margin = (chosen_rewards - rejected_rewards).mean()

        info = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "reward_margin": reward_margin.item(),
            "chosen_rewards_mean": chosen_rewards.mean().item(),
            "rejected_rewards_mean": rejected_rewards.mean().item(),
            "chosen_log_probs_mean": chosen_log_probs.mean().item(),
            "rejected_log_probs_mean": rejected_log_probs.mean().item(),
            "chosen_log_ratio_mean": chosen_log_ratio.mean().item(),
            "rejected_log_ratio_mean": rejected_log_ratio.mean().item(),
        }

    return loss, info


def compute_dpo_loss_with_kl(
    chosen_log_probs: torch.Tensor,
    rejected_log_probs: torch.Tensor,
    chosen_ref_log_probs: torch.Tensor,
    rejected_ref_log_probs: torch.Tensor,
    beta: float = 0.1,
    kl_coef: float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute DPO loss with explicit KL penalty.

    This variant adds an explicit KL divergence term to keep the policy
    close to the reference model.

    Args:
        chosen_log_probs: Log probs from policy for chosen responses
        rejected_log_probs: Log probs from policy for rejected responses
        chosen_ref_log_probs: Log probs from reference for chosen responses
        rejected_ref_log_probs: Log probs from reference for rejected responses
        beta: Temperature parameter
        kl_coef: Coefficient for explicit KL penalty

    Returns:
        Tuple of (total_loss, info_dict)
    """
    # Standard DPO loss
    dpo_loss, info = compute_dpo_loss(
        chosen_log_probs=chosen_log_probs,
        rejected_log_probs=rejected_log_probs,
        chosen_ref_log_probs=chosen_ref_log_probs,
        rejected_ref_log_probs=rejected_ref_log_probs,
        beta=beta,
    )

    # Compute KL divergence
    if kl_coef > 0:
        # KL(π || π_ref) ≈ log(π_ref) - log(π)
        kl_chosen = (chosen_ref_log_probs - chosen_log_probs).mean()
        kl_rejected = (rejected_ref_log_probs - rejected_log_probs).mean()
        kl_penalty = kl_coef * (kl_chosen + kl_rejected) / 2

        total_loss = dpo_loss + kl_penalty
        info["kl_penalty"] = kl_penalty.item()
        info["total_loss"] = total_loss.item()
    else:
        total_loss = dpo_loss
        info["kl_penalty"] = 0.0
        info["total_loss"] = dpo_loss.item()

    return total_loss, info


def compute_conservative_dpo_loss(
    chosen_log_probs: torch.Tensor,
    rejected_log_probs: torch.Tensor,
    chosen_ref_log_probs: torch.Tensor,
    rejected_ref_log_probs: torch.Tensor,
    beta: float = 0.1,
    margin: float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Conservative DPO loss with margin.

    This variant only penalizes when the model doesn't prefer chosen
    over rejected by at least a margin.

    Args:
        chosen_log_probs: Log probs from policy for chosen responses
        rejected_log_probs: Log probs from policy for rejected responses
        chosen_ref_log_probs: Log probs from reference for chosen responses
        rejected_ref_log_probs: Log probs from reference for rejected responses
        beta: Temperature parameter
        margin: Minimum margin required

    Returns:
        Tuple of (loss, info_dict)
    """
    # Compute log ratios
    chosen_log_ratio, rejected_log_ratio = compute_preference_log_ratio(
        chosen_log_probs=chosen_log_probs,
        rejected_log_probs=rejected_log_probs,
        chosen_ref_log_probs=chosen_ref_log_probs,
        rejected_ref_log_probs=rejected_ref_log_probs,
    )

    # Logits with margin
    logits = beta * (chosen_log_ratio - rejected_log_ratio - margin)

    # Loss: only penalize if margin not satisfied
    loss = -F.logsigmoid(logits).mean()

    # Statistics
    with torch.no_grad():
        accuracy = (logits > 0).float().mean()
        reward_margin = beta * (chosen_log_ratio - rejected_log_ratio).mean()

        info = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "reward_margin": reward_margin.item(),
            "margin": margin,
        }

    return loss, info
