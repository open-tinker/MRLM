"""
Trajectory dataset for SFT (Supervised Fine-Tuning).

Stores trajectories from environment interactions for world model training.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import random
import json
from pathlib import Path

from mrlm.core.types import Message, Observation, Reward


@dataclass
class Trajectory:
    """
    A trajectory from environment interaction.

    Contains the sequence of observations, actions, and rewards
    for training world models or behavioral cloning.

    Attributes:
        observations: List of observations
        actions: List of action messages
        rewards: List of rewards
        metadata: Additional metadata (environment type, task info, etc.)
    """

    observations: List[Observation]
    actions: List[Message]
    rewards: List[Reward]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Length is number of transitions."""
        return len(self.actions)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "observations": [
                {
                    "messages": [
                        {"role": m.role.value if hasattr(m.role, "value") else m.role, "content": m.content}
                        for m in obs.messages
                    ],
                    "done": obs.done,
                }
                for obs in self.observations
            ],
            "actions": [
                {"role": a.role.value if hasattr(a.role, "value") else a.role, "content": a.content}
                for a in self.actions
            ],
            "rewards": [{"value": r.value} for r in self.rewards],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trajectory":
        """Create from dictionary."""
        from mrlm.core.types import MessageRole

        observations = []
        for obs_data in data["observations"]:
            messages = [
                Message(role=m["role"], content=m["content"])
                for m in obs_data["messages"]
            ]
            observations.append(
                Observation(messages=messages, done=obs_data["done"])
            )

        actions = [
            Message(role=a["role"], content=a["content"])
            for a in data["actions"]
        ]

        rewards = [Reward(value=r["value"]) for r in data["rewards"]]

        return cls(
            observations=observations,
            actions=actions,
            rewards=rewards,
            metadata=data.get("metadata", {}),
        )


class TrajectoryDataset:
    """
    Dataset of trajectories for SFT training.

    Supports:
    - Behavioral cloning (state -> action)
    - Next state prediction (state + action -> next state)
    - Reward prediction (state + action -> reward)
    """

    def __init__(self, trajectories: Optional[List[Trajectory]] = None):
        """
        Initialize trajectory dataset.

        Args:
            trajectories: Optional initial trajectories
        """
        self.trajectories = trajectories or []

    def __len__(self) -> int:
        """Total number of trajectories."""
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Trajectory:
        """Get trajectory by index."""
        return self.trajectories[idx]

    def add_trajectory(self, trajectory: Trajectory):
        """Add a trajectory to the dataset."""
        self.trajectories.append(trajectory)

    def num_transitions(self) -> int:
        """Total number of transitions across all trajectories."""
        return sum(len(traj) for traj in self.trajectories)

    def save(self, path: Path):
        """
        Save dataset to JSON file.

        Args:
            path: Output path
        """
        data = {
            "trajectories": [traj.to_dict() for traj in self.trajectories],
            "num_trajectories": len(self.trajectories),
            "num_transitions": self.num_transitions(),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TrajectoryDataset":
        """
        Load dataset from JSON file.

        Args:
            path: Input path

        Returns:
            TrajectoryDataset
        """
        with open(path, "r") as f:
            data = json.load(f)

        trajectories = [
            Trajectory.from_dict(traj_data)
            for traj_data in data["trajectories"]
        ]

        return cls(trajectories=trajectories)

    def split(self, train_ratio: float = 0.8) -> tuple["TrajectoryDataset", "TrajectoryDataset"]:
        """
        Split dataset into train and validation sets.

        Args:
            train_ratio: Ratio of data for training

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        indices = list(range(len(self.trajectories)))
        random.shuffle(indices)

        split_idx = int(len(indices) * train_ratio)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        train_trajs = [self.trajectories[i] for i in train_indices]
        val_trajs = [self.trajectories[i] for i in val_indices]

        return TrajectoryDataset(train_trajs), TrajectoryDataset(val_trajs)

    def get_batch(self, batch_size: int, shuffle: bool = True) -> List[Trajectory]:
        """
        Get a batch of trajectories.

        Args:
            batch_size: Number of trajectories to sample
            shuffle: Whether to shuffle before sampling

        Returns:
            List of trajectories
        """
        if shuffle:
            indices = random.sample(range(len(self.trajectories)), min(batch_size, len(self.trajectories)))
        else:
            indices = list(range(min(batch_size, len(self.trajectories))))

        return [self.trajectories[i] for i in indices]

    def get_transitions(self) -> List[tuple[Observation, Message, Observation, Reward]]:
        """
        Get all transitions (s, a, s', r) from trajectories.

        Returns:
            List of (observation, action, next_observation, reward) tuples
        """
        transitions = []

        for traj in self.trajectories:
            for i in range(len(traj)):
                obs = traj.observations[i]
                action = traj.actions[i]
                next_obs = traj.observations[i + 1] if i + 1 < len(traj.observations) else None
                reward = traj.rewards[i]

                if next_obs is not None:
                    transitions.append((obs, action, next_obs, reward))

        return transitions

    def collect_from_environment(
        self,
        env,
        policy,
        num_trajectories: int,
        max_steps: int = 100,
    ) -> int:
        """
        Collect trajectories from environment using a policy.

        Args:
            env: Environment to collect from
            policy: Policy function (observation -> action)
            num_trajectories: Number of trajectories to collect
            max_steps: Maximum steps per trajectory

        Returns:
            Number of trajectories collected
        """
        collected = 0

        for _ in range(num_trajectories):
            observations = []
            actions = []
            rewards = []

            obs = env.reset()
            observations.append(obs)

            for step in range(max_steps):
                # Get action from policy
                action = policy(obs)
                actions.append(action)

                # Take step
                next_obs, reward = env.step(action)
                observations.append(next_obs)
                rewards.append(reward)

                if next_obs.done:
                    break

                obs = next_obs

            # Create trajectory
            trajectory = Trajectory(
                observations=observations,
                actions=actions,
                rewards=rewards,
                metadata={
                    "num_steps": len(actions),
                    "total_reward": sum(r.value for r in rewards),
                },
            )

            self.add_trajectory(trajectory)
            collected += 1

        return collected

    def filter_by_reward(self, min_reward: Optional[float] = None, max_reward: Optional[float] = None):
        """
        Filter trajectories by total reward.

        Args:
            min_reward: Minimum total reward (inclusive)
            max_reward: Maximum total reward (inclusive)

        Returns:
            New TrajectoryDataset with filtered trajectories
        """
        filtered = []

        for traj in self.trajectories:
            total_reward = sum(r.value for r in traj.rewards)

            if min_reward is not None and total_reward < min_reward:
                continue
            if max_reward is not None and total_reward > max_reward:
                continue

            filtered.append(traj)

        return TrajectoryDataset(trajectories=filtered)
