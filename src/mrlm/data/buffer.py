"""
Rollout buffer for storing and managing RL training data.

This module provides the RolloutBuffer class for collecting trajectories
during environment interaction and preparing them for training.
"""

from typing import List, Optional

import torch
from torch.utils.data import Dataset

from mrlm.core.types import Message, Observation, Reward, RolloutBatch


class RolloutBuffer:
    """
    Buffer for storing rollout data during RL training.

    This buffer stores trajectories collected from environment interactions
    and provides utilities for computing returns, advantages, and converting
    to PyTorch datasets for training.

    Attributes:
        observations: List of observations
        actions: List of actions (as Messages)
        rewards: List of rewards
        values: Optional value estimates
        log_probs: Optional log probabilities
        dones: Episode termination flags

    Example:
        >>> buffer = RolloutBuffer()
        >>> buffer.add(obs, action, reward, log_prob, value, done)
        >>> buffer.compute_returns_and_advantages(gamma=0.99, gae_lambda=0.95)
        >>> dataset = buffer.to_dataset()
    """

    def __init__(self, buffer_size: Optional[int] = None):
        """
        Initialize rollout buffer.

        Args:
            buffer_size: Maximum buffer size (None for unlimited)
        """
        self.buffer_size = buffer_size

        # Storage
        self.observations: List[Observation] = []
        self.actions: List[Message] = []
        self.rewards: List[Reward] = []
        self.values: List[float] = []
        self.log_probs: List[torch.Tensor] = []
        self.dones: List[bool] = []

        # Computed values
        self.returns: Optional[torch.Tensor] = None
        self.advantages: Optional[torch.Tensor] = None

    def add(
        self,
        observation: Observation,
        action: Message,
        reward: Reward,
        log_prob: Optional[torch.Tensor] = None,
        value: Optional[float] = None,
        done: bool = False,
    ):
        """
        Add a transition to the buffer.

        Args:
            observation: Observation from environment
            action: Action taken
            reward: Reward received
            log_prob: Log probability of action (for PPO)
            value: Value estimate (for PPO)
            done: Whether episode terminated
        """
        # Check buffer size
        if self.buffer_size is not None and len(self) >= self.buffer_size:
            # Remove oldest entry (FIFO)
            self.observations.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            if self.values:
                self.values.pop(0)
            if self.log_probs:
                self.log_probs.pop(0)
            if self.dones:
                self.dones.pop(0)

        # Add new entry
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)

        if value is not None:
            self.values.append(value)
        if log_prob is not None:
            self.log_probs.append(log_prob)
        if done is not None:
            self.dones.append(done)

    def compute_returns(self, gamma: float = 0.99) -> torch.Tensor:
        """
        Compute discounted returns.

        Returns = Σ(γ^t * reward_t)

        Args:
            gamma: Discount factor

        Returns:
            Tensor of returns for each timestep
        """
        returns = []
        R = 0.0

        # Work backwards through rewards
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                R = 0.0  # Reset at episode boundaries
            R = reward.value + gamma * R
            returns.insert(0, R)

        self.returns = torch.tensor(returns, dtype=torch.float32)
        return self.returns

    def compute_gae(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).

        GAE balances bias and variance in advantage estimation.

        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter (0 = high bias, 1 = high variance)
            normalize: Whether to normalize advantages

        Returns:
            Tuple of (advantages, returns)

        References:
            Schulman et al., "High-Dimensional Continuous Control Using
            Generalized Advantage Estimation", 2016
        """
        if not self.values:
            raise ValueError("Cannot compute GAE without value estimates")

        advantages = []
        gae = 0.0

        # Convert to tensors
        values_tensor = torch.tensor(self.values, dtype=torch.float32)
        rewards_tensor = torch.tensor(
            [r.value for r in self.rewards], dtype=torch.float32
        )
        dones_tensor = torch.tensor(self.dones, dtype=torch.float32)

        # Add final value of 0 (assuming episode ends)
        values_extended = torch.cat([values_tensor, torch.tensor([0.0])])

        # Compute advantages backwards
        for t in reversed(range(len(rewards_tensor))):
            # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = (
                rewards_tensor[t]
                + gamma * values_extended[t + 1] * (1 - dones_tensor[t])
                - values_extended[t]
            )

            # GAE: A_t = δ_t + γλ * A_{t+1}
            gae = delta + gamma * gae_lambda * (1 - dones_tensor[t]) * gae
            advantages.insert(0, gae.item())

        # Convert to tensor
        self.advantages = torch.tensor(advantages, dtype=torch.float32)

        # Normalize advantages (reduces variance)
        if normalize and len(self.advantages) > 1:
            self.advantages = (self.advantages - self.advantages.mean()) / (
                self.advantages.std() + 1e-8
            )

        # Returns = advantages + values
        self.returns = self.advantages + values_tensor

        return self.advantages, self.returns

    def to_batch(self) -> RolloutBatch:
        """
        Convert buffer to RolloutBatch.

        Returns:
            RolloutBatch containing all data
        """
        return RolloutBatch(
            observations=self.observations.copy(),
            actions=self.actions.copy(),
            rewards=self.rewards.copy(),
            values=torch.tensor(self.values) if self.values else None,
            log_probs=torch.stack(self.log_probs) if self.log_probs else None,
            advantages=self.advantages,
            returns=self.returns,
            dones=torch.tensor(self.dones) if self.dones else None,
        )

    def to_dataset(self) -> "RolloutDataset":
        """
        Convert buffer to PyTorch Dataset.

        Returns:
            RolloutDataset for use with DataLoader
        """
        return RolloutDataset(self.to_batch())

    def clear(self):
        """Clear all data from buffer."""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.returns = None
        self.advantages = None

    def __len__(self) -> int:
        """Return number of transitions in buffer."""
        return len(self.observations)

    def __repr__(self) -> str:
        """String representation."""
        return f"RolloutBuffer(size={len(self)}, has_values={bool(self.values)})"


class RolloutDataset(Dataset):
    """
    PyTorch Dataset wrapper for rollout data.

    This allows rollout data to be used with PyTorch DataLoader for
    batched training.

    Example:
        >>> dataset = RolloutDataset(rollout_batch)
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for batch in dataloader:
        ...     # Train on batch
        ...     pass
    """

    def __init__(self, batch: RolloutBatch):
        """
        Initialize dataset from RolloutBatch.

        Args:
            batch: RolloutBatch containing rollout data
        """
        self.batch = batch

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.batch)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample.

        Args:
            idx: Index of sample

        Returns:
            Dictionary containing sample data
        """
        sample = {
            "observation": self.batch.observations[idx],
            "action": self.batch.actions[idx],
            "reward": self.batch.rewards[idx],
        }

        if self.batch.values is not None:
            sample["value"] = self.batch.values[idx]

        if self.batch.log_probs is not None:
            sample["log_prob"] = self.batch.log_probs[idx]

        if self.batch.advantages is not None:
            sample["advantage"] = self.batch.advantages[idx]

        if self.batch.returns is not None:
            sample["return"] = self.batch.returns[idx]

        if self.batch.dones is not None:
            sample["done"] = self.batch.dones[idx]

        return sample
