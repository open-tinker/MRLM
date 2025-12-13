"""
MRLM: Multi-Agent Reinforcement Learning Library for LLMs

A comprehensive library for multi-agent reinforcement learning training of Large Language Models.
Features server-client architecture, distributed training, and multiple RL algorithms (PPO, GRPO, DPO, SFT).
"""

__version__ = "0.1.0"

# Core types
from mrlm.core.types import (
    EnvironmentMode,
    Message,
    MessageRole,
    Observation,
    Reward,
    RolloutBatch,
)

# Core classes
from mrlm.core.base import BaseEnvironment
from mrlm.core.simulated_environment import SimulatedEnvironment
from mrlm.core.llm_environment import LLMEnvironment

# Algorithms
from mrlm.algorithms.base_trainer import BaseTrainer
from mrlm.algorithms.ppo import PPOTrainer
from mrlm.algorithms.grpo import GRPOTrainer
from mrlm.algorithms.dpo import DPOTrainer
from mrlm.algorithms.sft import SFTTrainer

# Configuration
from mrlm.config import (
    ExperimentConfig,
    TrainingConfig,
    ModelConfig,
    PPOConfig,
    GRPOConfig,
    DPOConfig,
    SFTConfig,
)

__all__ = [
    # Version
    "__version__",
    # Core types
    "EnvironmentMode",
    "Message",
    "MessageRole",
    "Observation",
    "Reward",
    "RolloutBatch",
    # Core classes
    "BaseEnvironment",
    "SimulatedEnvironment",
    "BaseTrainer",
    "LLMEnvironment",
    # Algorithms
    "PPOTrainer",
    "GRPOTrainer",
    "DPOTrainer",
    "SFTTrainer",
    # Configuration
    "ExperimentConfig",
    "TrainingConfig",
    "ModelConfig",
    "PPOConfig",
    "GRPOConfig",
    "DPOConfig",
    "SFTConfig",
]
