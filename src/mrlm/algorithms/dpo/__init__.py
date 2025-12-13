"""
DPO (Direct Preference Optimization) implementation.

DPO directly optimizes preferences without requiring online RL rollouts.
"""

from mrlm.algorithms.dpo.loss import (
    compute_dpo_loss,
    compute_preference_log_ratio,
)
from mrlm.algorithms.dpo.trainer import DPOTrainer
from mrlm.algorithms.dpo.dataset import PreferenceDataset, PreferencePair

__all__ = [
    "compute_dpo_loss",
    "compute_preference_log_ratio",
    "DPOTrainer",
    "PreferenceDataset",
    "PreferencePair",
]
