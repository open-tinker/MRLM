"""
SFT (Supervised Fine-Tuning) trainer for world model training.

Trains LLM to predict future states from environment trajectories.
Supports distributed training with FSDP/DDP.
"""

from typing import Dict, List, Optional
from pathlib import Path

import torch
from torch.optim import Optimizer, AdamW
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm

from mrlm.core.base import BaseEnvironment
from mrlm.core.types import Message, MessageRole
from mrlm.algorithms.sft.dataset import TrajectoryDataset, Trajectory
from mrlm.algorithms.sft.loss import (
    compute_behavioral_cloning_loss,
    compute_next_state_loss,
    compute_combined_sft_loss,
)
from mrlm.algorithms.base_trainer import BaseTrainer
from mrlm.config.training_config import ExperimentConfig


class SFTTrainer(BaseTrainer):
    """
    SFT trainer for world model training and behavioral cloning.

    Can be used for:
    1. Behavioral cloning: Learn policy from demonstrations
    2. World model: Learn to predict next states
    3. Pre-training before RL fine-tuning
    4. Regularization alongside RL training
    """

    def __init__(
        self,
        policy_env: BaseEnvironment,
        eval_envs: List[BaseEnvironment],
        config: ExperimentConfig,
        trajectory_dataset: Optional[TrajectoryDataset] = None,
        optimizer: Optional[Optimizer] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize SFT trainer.

        Args:
            policy_env: LLM environment in SERVER mode (for training)
            eval_envs: List of evaluation environments
            config: Experiment configuration
            trajectory_dataset: Dataset of trajectories (collected if None)
            optimizer: Optional optimizer (created if not provided)
            device: Device for training
        """
        super().__init__(policy_env, eval_envs, config, device)

        # SFT-specific config
        self.sft_config = config.sft
        if self.sft_config is None:
            raise ValueError("SFT config must be provided for SFTTrainer")

        # Get model and tokenizer
        if hasattr(policy_env, "model"):
            self.model: PreTrainedModel = policy_env.model
            self.tokenizer: PreTrainedTokenizer = policy_env.tokenizer
        else:
            raise ValueError("Policy environment must have 'model' and 'tokenizer' attributes")

        # Trajectory dataset
        self.trajectory_dataset = trajectory_dataset or TrajectoryDataset()

        # Create optimizer if needed
        if optimizer is None:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )
        else:
            self.optimizer = optimizer

    def collect_rollouts(self) -> TrajectoryDataset:
        """
        Collect trajectories from environments.

        Returns:
            TrajectoryDataset with new trajectories
        """
        dataset = TrajectoryDataset()
        total_episodes = 0
        target_episodes = self.config.training.episodes_per_iteration

        self.model.eval()
        with torch.no_grad():
            while total_episodes < target_episodes:
                # Select random environment
                import random
                env = random.choice(self.eval_envs)

                # Collect one trajectory
                observations = []
                actions = []
                rewards = []

                obs = env.reset()
                observations.append(obs)

                episode_length = 0
                done = False

                while not done and episode_length < self.config.training.max_episode_length:
                    # Generate action
                    from mrlm.models.generation import generate_response

                    response = generate_response(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        messages=obs.messages,
                        generation_config=self.config.model.generation.to_generation_config(),
                        device=self.device,
                    )

                    action = Message(role=MessageRole.ASSISTANT, content=response)
                    actions.append(action)

                    # Take step
                    next_obs, reward = env.step(action)
                    observations.append(next_obs)
                    rewards.append(reward)

                    done = next_obs.done
                    obs = next_obs
                    episode_length += 1

                # Create trajectory
                trajectory = Trajectory(
                    observations=observations,
                    actions=actions,
                    rewards=rewards,
                    metadata={
                        "num_steps": episode_length,
                        "total_reward": sum(r.value for r in rewards),
                    },
                )

                dataset.add_trajectory(trajectory)
                total_episodes += 1

        return dataset

    def train_epoch(self, rollouts=None) -> Dict[str, float]:
        """
        Train for one epoch on collected trajectories.

        Args:
            rollouts: Not used (trajectories are in self.trajectory_dataset)

        Returns:
            Dictionary of training metrics
        """
        if len(self.trajectory_dataset) == 0:
            return {}

        # Get all transitions
        transitions = self.trajectory_dataset.get_transitions()
        if not transitions:
            return {}

        # Shuffle transitions
        import random
        random.shuffle(transitions)

        metrics = {
            "bc_loss": 0.0,
            "world_model_loss": 0.0,
            "total_loss": 0.0,
        }

        num_updates = 0
        batch_size = self.config.training.batch_size

        # Train in batches
        for start_idx in range(0, len(transitions), batch_size):
            end_idx = min(start_idx + batch_size, len(transitions))
            batch = transitions[start_idx:end_idx]

            # Extract batch data
            observations = []
            actions = []
            next_observations = []

            for obs, action, next_obs, reward in batch:
                # Extract last message from observation
                if obs.messages:
                    observations.append(obs.messages[-1])
                else:
                    continue

                actions.append(action)

                # Create message from next observation
                if next_obs.messages:
                    next_observations.append(next_obs.messages[-1])
                else:
                    # Create a message describing the next state
                    next_msg = Message(
                        role=MessageRole.SYSTEM,
                        content=f"Done: {next_obs.done}",
                    )
                    next_observations.append(next_msg)

            if not observations:
                continue

            # Set model to training mode
            self.model.train()

            # Compute loss based on mode
            if self.sft_config.mode == "behavioral_cloning":
                loss, info = compute_behavioral_cloning_loss(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    observations=observations,
                    actions=actions,
                    device=self.device,
                )
                metrics["bc_loss"] += info["bc_loss"]

            elif self.sft_config.mode == "world_model":
                loss, info = compute_next_state_loss(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    observations=observations,
                    actions=actions,
                    next_observations=next_observations,
                    device=self.device,
                )
                metrics["world_model_loss"] += info["next_state_loss"]

            elif self.sft_config.mode == "combined":
                loss, info = compute_combined_sft_loss(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    observations=observations,
                    actions=actions,
                    next_observations=next_observations,
                    device=self.device,
                    bc_weight=self.sft_config.bc_weight,
                    world_model_weight=self.sft_config.world_model_weight,
                )
                metrics["bc_loss"] += info["bc_loss"]
                metrics["world_model_loss"] += info["world_model_loss"]

            else:
                raise ValueError(f"Unknown SFT mode: {self.sft_config.mode}")

            metrics["total_loss"] += loss.item()

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            if self.config.training.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.max_grad_norm,
                )

            self.optimizer.step()
            num_updates += 1

        # Average metrics
        if num_updates > 0:
            for key in metrics:
                metrics[key] /= num_updates

        return metrics

    def train(
        self,
        num_iterations: int,
        collect_every: int = 1,
        eval_every: int = 10,
        save_every: int = 10,
    ):
        """
        Train the model using SFT.

        Args:
            num_iterations: Number of training iterations
            collect_every: Collect new trajectories every N iterations
            eval_every: Evaluate every N iterations
            save_every: Save checkpoint every N iterations
        """
        for iteration in tqdm(range(num_iterations), desc="SFT Training"):
            # Collect trajectories
            if iteration % collect_every == 0 or len(self.trajectory_dataset) == 0:
                print(f"\nCollecting trajectories (iteration {iteration})...")
                new_trajectories = self.collect_rollouts()
                print(f"Collected {len(new_trajectories)} trajectories")

                # Add to dataset
                for traj in new_trajectories.trajectories:
                    self.trajectory_dataset.add_trajectory(traj)

                # Optional: Filter to keep only high-reward trajectories
                if self.sft_config.filter_low_reward:
                    original_size = len(self.trajectory_dataset)
                    self.trajectory_dataset = self.trajectory_dataset.filter_by_reward(
                        min_reward=self.sft_config.min_reward_threshold
                    )
                    print(f"Filtered dataset: {original_size} -> {len(self.trajectory_dataset)} trajectories")

            # Train for one epoch
            train_metrics = self.train_epoch()

            # Log metrics
            if iteration % 10 == 0:
                metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in train_metrics.items())
                print(f"Iteration {iteration}: {metrics_str}")

            # Evaluate
            if iteration % eval_every == 0:
                eval_metrics = self.evaluate()
                if eval_metrics:
                    eval_str = ", ".join(f"{k}: {v:.4f}" for k, v in eval_metrics.items())
                    print(f"Evaluation: {eval_str}")

            # Save checkpoint
            if iteration % save_every == 0:
                self.save_checkpoint(f"iteration_{iteration}")

        # Final save
        self.save_checkpoint("final")

        # Save trajectory dataset
        dataset_path = Path(self.config.output.save_dir) / "trajectory_dataset.json"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self.trajectory_dataset.save(dataset_path)
        print(f"Saved trajectory dataset to {dataset_path}")

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the current policy.

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_reward = 0.0
        num_episodes = 0

        with torch.no_grad():
            for env in self.eval_envs[:min(3, len(self.eval_envs))]:  # Sample few envs
                obs = env.reset()
                episode_reward = 0.0
                episode_length = 0
                done = False

                while not done and episode_length < self.config.training.max_episode_length:
                    # Generate action
                    from mrlm.models.generation import generate_response

                    response = generate_response(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        messages=obs.messages,
                        generation_config=self.config.model.generation.to_generation_config(),
                        device=self.device,
                    )

                    action = Message(role=MessageRole.ASSISTANT, content=response)
                    next_obs, reward = env.step(action)

                    episode_reward += reward.value
                    episode_length += 1
                    done = next_obs.done
                    obs = next_obs

                total_reward += episode_reward
                num_episodes += 1

        avg_reward = total_reward / max(num_episodes, 1)
        return {"eval/avg_reward": avg_reward, "eval/num_episodes": num_episodes}
