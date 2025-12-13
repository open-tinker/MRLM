"""
Base trainer class for RL algorithms.

This module provides the BaseTrainer abstract class that all RL algorithm
trainers (PPO, GRPO, DPO) inherit from.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from mrlm.core.base import BaseEnvironment
from mrlm.core.llm_environment import LLMEnvironment
from mrlm.data.buffer import RolloutBuffer

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """
    Abstract base class for all RL trainers.

    This class defines the common interface and functionality for all RL
    algorithms (PPO, GRPO, DPO). Subclasses must implement algorithm-specific
    methods for rollout collection, loss computation, and training steps.

    Attributes:
        policy_env: LLMEnvironment being trained (in SERVER mode)
        eval_envs: List of environments for evaluation
        device: Device for training (cuda/cpu)
        global_step: Current training step
        epoch: Current epoch

    Example:
        >>> class MyTrainer(BaseTrainer):
        ...     def collect_rollouts(self):
        ...         # Implement rollout collection
        ...         pass
        ...     def train_step(self, batch):
        ...         # Implement training step
        ...         pass
    """

    def __init__(
        self,
        policy_env: LLMEnvironment,
        eval_envs: List[BaseEnvironment],
        learning_rate: float = 3e-4,
        max_grad_norm: float = 0.5,
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Initialize base trainer.

        Args:
            policy_env: LLM environment to train (must be in SERVER mode)
            eval_envs: List of environments for evaluation
            learning_rate: Learning rate for optimizer
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to use (if None, auto-detect)
            checkpoint_dir: Directory for saving checkpoints
        """
        self.policy_env = policy_env
        self.eval_envs = eval_envs
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

        # Verify policy environment is in server mode
        if not policy_env.is_server_mode:
            logger.warning("Policy environment not in SERVER mode, switching...")
            policy_env.set_mode(EnvironmentMode.SERVER)

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        # Move model to device
        self.policy_env.model = self.policy_env.model.to(self.device)

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("./checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer (subclass can override)
        self.optimizer = self._build_optimizer()
        self.scheduler: Optional[_LRScheduler] = None

        # Metrics storage
        self.metrics_history: List[Dict[str, float]] = []

    def _build_optimizer(self) -> Optimizer:
        """
        Build optimizer for training.

        Override this method to use a different optimizer.

        Returns:
            Optimizer instance
        """
        return torch.optim.AdamW(
            self.policy_env.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
        )

    @abstractmethod
    def collect_rollouts(self) -> RolloutBuffer:
        """
        Collect rollout data from environments.

        This method should:
        1. Run policy in environments to collect trajectories
        2. Store data in RolloutBuffer
        3. Compute advantages/returns as needed

        Returns:
            RolloutBuffer containing collected data

        Example:
            >>> def collect_rollouts(self):
            ...     buffer = RolloutBuffer()
            ...     for env in self.eval_envs:
            ...         obs = env.reset()
            ...         # Collect trajectory...
            ...         buffer.add(obs, action, reward, ...)
            ...     return buffer
        """
        pass

    @abstractmethod
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """
        Perform single training step on a batch.

        This method should:
        1. Forward pass through model
        2. Compute loss
        3. Backward pass
        4. Optimizer step
        5. Return metrics

        Args:
            batch: Batch of training data

        Returns:
            Dictionary of training metrics

        Example:
            >>> def train_step(self, batch):
            ...     loss = self.compute_loss(batch)
            ...     loss.backward()
            ...     self.optimizer.step()
            ...     return {"loss": loss.item()}
        """
        pass

    def train_epoch(
        self, rollouts: RolloutBuffer, num_epochs: int = 4, batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Train for multiple epochs on rollout data.

        Args:
            rollouts: Collected rollout data
            num_epochs: Number of epochs to train
            batch_size: Batch size for training

        Returns:
            Dictionary of aggregated metrics
        """
        from torch.utils.data import DataLoader

        dataset = rollouts.to_dataset()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        all_metrics = []

        for epoch_idx in range(num_epochs):
            epoch_metrics = []

            for batch in dataloader:
                metrics = self.train_step(batch)
                epoch_metrics.append(metrics)
                self.global_step += 1

            # Average metrics for this epoch
            avg_metrics = self._aggregate_metrics(epoch_metrics)
            avg_metrics["epoch"] = epoch_idx
            all_metrics.append(avg_metrics)

        # Average across all epochs
        final_metrics = self._aggregate_metrics(all_metrics)
        return final_metrics

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate policy on evaluation environments.

        Returns:
            Dictionary of evaluation metrics
        """
        self.policy_env.model.eval()

        eval_rewards = []
        eval_lengths = []

        with torch.no_grad():
            for env in self.eval_envs:
                obs = env.reset()
                episode_reward = 0.0
                episode_length = 0
                done = False

                while not done and episode_length < 512:  # Max length
                    # Generate action from policy
                    from mrlm.core.types import Message, MessageRole

                    # Use last message as context
                    if obs.messages:
                        # Environment provides messages, continue conversation
                        action_content = self._generate_action(obs.messages)
                    else:
                        # Start new conversation
                        action_content = "Hello"

                    action = Message(role=MessageRole.ASSISTANT, content=action_content)

                    obs, reward = env.step(action)
                    episode_reward += reward.value
                    episode_length += 1
                    done = obs.done

                eval_rewards.append(episode_reward)
                eval_lengths.append(episode_length)

        self.policy_env.model.train()

        return {
            "eval/mean_reward": sum(eval_rewards) / len(eval_rewards) if eval_rewards else 0.0,
            "eval/max_reward": max(eval_rewards) if eval_rewards else 0.0,
            "eval/min_reward": min(eval_rewards) if eval_rewards else 0.0,
            "eval/mean_length": sum(eval_lengths) / len(eval_lengths) if eval_lengths else 0.0,
        }

    def _generate_action(self, messages: List) -> str:
        """Generate action from current policy (simple version for eval)."""
        from mrlm.models.generation import generate_response

        response = generate_response(
            model=self.policy_env.model,
            tokenizer=self.policy_env.tokenizer,
            messages=messages,
            generation_config=self.policy_env.generation_config,
            device=self.device,
        )
        return response

    def train(
        self, num_iterations: int, eval_every: int = 10, save_every: int = 10
    ):
        """
        Main training loop.

        Args:
            num_iterations: Number of training iterations
            eval_every: Evaluate every N iterations
            save_every: Save checkpoint every N iterations
        """
        logger.info(f"Starting training for {num_iterations} iterations")

        for iteration in tqdm(range(num_iterations), desc="Training"):
            self.epoch = iteration

            # Collect rollouts
            logger.info(f"Iteration {iteration}: Collecting rollouts...")
            rollouts = self.collect_rollouts()
            logger.info(f"Collected {len(rollouts)} transitions")

            # Train on collected data
            logger.info("Training on rollouts...")
            train_metrics = self.train_epoch(rollouts)

            # Evaluate
            if iteration % eval_every == 0:
                logger.info("Evaluating...")
                eval_metrics = self.evaluate()
                train_metrics.update(eval_metrics)

            # Log metrics
            train_metrics["iteration"] = iteration
            self.metrics_history.append(train_metrics)
            self._log_metrics(train_metrics)

            # Save checkpoint
            if iteration % save_every == 0:
                self.save_checkpoint(f"iteration_{iteration}")

        logger.info("Training complete!")

    def save_checkpoint(self, name: str = "checkpoint"):
        """
        Save model checkpoint.

        Args:
            name: Name for checkpoint directory
        """
        checkpoint_path = self.checkpoint_dir / name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        self.policy_env.model.save_pretrained(checkpoint_path)
        self.policy_env.tokenizer.save_pretrained(checkpoint_path)

        # Save optimizer state
        torch.save(
            {
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "epoch": self.epoch,
            },
            checkpoint_path / "trainer_state.pt",
        )

        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
        """
        # Load model and tokenizer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.policy_env.model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        self.policy_env.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.policy_env.model = self.policy_env.model.to(self.device)

        # Load optimizer state
        trainer_state = torch.load(checkpoint_path / "trainer_state.pt")
        self.optimizer.load_state_dict(trainer_state["optimizer_state_dict"])
        self.global_step = trainer_state["global_step"]
        self.epoch = trainer_state["epoch"]

        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate list of metric dictionaries."""
        if not metrics_list:
            return {}

        aggregated = {}
        keys = metrics_list[0].keys()

        for key in keys:
            values = [m[key] for m in metrics_list if key in m]
            if values:
                aggregated[key] = sum(values) / len(values)

        return aggregated

    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to console and/or tracking system."""
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        logger.info(f"Metrics: {metrics_str}")
        print(f"[Iter {self.epoch}] {metrics_str}")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}(step={self.global_step}, "
            f"epoch={self.epoch}, device={self.device})"
        )


# Import for type hints
from mrlm.core.types import EnvironmentMode
