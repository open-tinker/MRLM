"""
GRPO (Group Relative Policy Optimization) implementation.

GRPO normalizes rewards within groups of samples to reduce variance.
"""

from mrlm.algorithms.grpo.loss import (
    compute_grpo_loss,
    compute_group_advantages,
    normalize_rewards_by_group,
)
from mrlm.algorithms.grpo.trainer import GRPOTrainer

__all__ = [
    "compute_grpo_loss",
    "compute_group_advantages",
    "normalize_rewards_by_group",
    "GRPOTrainer",
]
