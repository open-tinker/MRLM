"""
SFT (Supervised Fine-Tuning) for world model training.

Train LLM to predict future states from environment trajectories.
Can be used as pre-training or regularization alongside RL.
"""

from mrlm.algorithms.sft.loss import (
    compute_sft_loss,
    compute_next_state_loss,
    compute_behavioral_cloning_loss,
)
from mrlm.algorithms.sft.dataset import TrajectoryDataset, Trajectory
from mrlm.algorithms.sft.trainer import SFTTrainer

__all__ = [
    "compute_sft_loss",
    "compute_next_state_loss",
    "compute_behavioral_cloning_loss",
    "TrajectoryDataset",
    "Trajectory",
    "SFTTrainer",
]
