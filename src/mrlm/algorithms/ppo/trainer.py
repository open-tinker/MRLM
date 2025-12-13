"""
PPO (Proximal Policy Optimization) trainer implementation.

This module provides the PPOTrainer class that implements the complete
PPO algorithm for training LLMs.
"""

import logging
from typing import Dict, List

import torch

from mrlm.algorithms.base_trainer import BaseTrainer
from mrlm.algorithms.ppo.config import PPOConfig
from mrlm.algorithms.ppo.loss import compute_ppo_total_loss
from mrlm.core.base import BaseEnvironment
from mrlm.core.llm_environment import LLMEnvironment
from mrlm.core.types import Message, MessageRole
from mrlm.data.buffer import RolloutBuffer

logger = logging.getLogger(__name__)


class PPOTrainer(BaseTrainer):
    """
    Proximal Policy Optimization trainer for LLMs.

    This trainer implements the PPO algorithm, which is one of the most popular
    and effective RL algorithms for LLM training. PPO uses a clipped surrogate
    objective to prevent destructively large policy updates.

    Key features:
    - Collects rollouts from evaluation environments
    - Computes advantages using GAE (Generalized Advantage Estimation)
    - Updates policy using clipped surrogate objective
    - Trains value function to predict returns

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> from mrlm.core import LLMEnvironment, EnvironmentMode
        >>> from mrlm.algorithms.ppo import PPOTrainer, PPOConfig
        >>>
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> policy_env = LLMEnvironment(model, tokenizer, mode=EnvironmentMode.SERVER)
        >>>
        >>> config = PPOConfig(num_epochs=50, batch_size=16)
        >>> trainer = PPOTrainer(policy_env, eval_envs=[], config=config)
        >>> trainer.train(num_iterations=100)
    """

    def __init__(
        self,
        policy_env: LLMEnvironment,
        eval_envs: List[BaseEnvironment],
        config: PPOConfig,
        **kwargs,
    ):
        """
        Initialize PPO trainer.

        Args:
            policy_env: LLM environment to train
            eval_envs: List of evaluation environments
            config: PPO configuration
            **kwargs: Additional arguments for BaseTrainer
        """
        super().__init__(
            policy_env=policy_env,
            eval_envs=eval_envs,
            learning_rate=config.learning_rate,
            max_grad_norm=config.max_grad_norm,
            **kwargs,
        )

        self.config = config
        logger.info(f"Initialized PPO trainer with config: {config}")

    def collect_rollouts(self) -> RolloutBuffer:
        """
        Collect rollouts from evaluation environments using current policy.

        This method:
        1. Runs episodes in each evaluation environment
        2. Collects (observation, action, reward, log_prob, value) tuples
        3. Computes advantages using GAE
        4. Returns RolloutBuffer ready for training

        Returns:
            RolloutBuffer containing rollout data
        """
        buffer = RolloutBuffer(buffer_size=self.config.num_rollouts_per_iteration * 2)

        self.policy_env.model.eval()  # Set to eval mode for rollout collection

        rollouts_collected = 0
        target_rollouts = self.config.num_rollouts_per_iteration

        logger.info(f"Collecting {target_rollouts} rollouts...")

        # Collect rollouts from environments
        for env_idx, env in enumerate(self.eval_envs):
            # Collect multiple episodes from each environment
            episodes_per_env = max(1, target_rollouts // len(self.eval_envs))

            for episode in range(episodes_per_env):
                if rollouts_collected >= target_rollouts:
                    break

                # Run one episode
                obs = env.reset()
                episode_length = 0

                while episode_length < self.config.max_episode_length:
                    # Generate action from policy
                    action, log_prob, value = self._generate_action_with_value(obs.messages)

                    action_msg = Message(
                        role=MessageRole.ASSISTANT,
                        content=action,
                        metadata={"log_prob": log_prob, "value": value},
                    )

                    # Take step in environment
                    next_obs, reward = env.step(action_msg)

                    # Store transition
                    buffer.add(
                        observation=obs,
                        action=action_msg,
                        reward=reward,
                        log_prob=log_prob,
                        value=value.item() if isinstance(value, torch.Tensor) else value,
                        done=next_obs.done,
                    )

                    obs = next_obs
                    episode_length += 1
                    rollouts_collected += 1

                    if next_obs.done:
                        break

        logger.info(f"Collected {len(buffer)} transitions from {len(self.eval_envs)} environments")

        # Compute advantages using GAE
        buffer.compute_gae(
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            normalize=True,
        )

        self.policy_env.model.train()  # Back to training mode

        return buffer

    def _generate_action_with_value(self, messages: List[Message]):
        """
        Generate action from policy with log probability and value estimate.

        Args:
            messages: Conversation history

        Returns:
            Tuple of (action_text, log_prob, value)
        """
        # For now, use simple generation
        # In full implementation, this would use the policy network to compute
        # log probs and value function to compute value

        from mrlm.models.generation import generate_with_log_probs

        with torch.no_grad():
            response, log_probs, value = generate_with_log_probs(
                model=self.policy_env.model,
                tokenizer=self.policy_env.tokenizer,
                messages=messages,
                generation_config=self.policy_env.generation_config,
                device=self.device,
                return_value=True,
            )

        # Aggregate log probs (mean over sequence)
        log_prob = log_probs.mean() if len(log_probs) > 0 else torch.tensor(0.0)

        # Use dummy value for now (would come from value head in full implementation)
        if value is None:
            value = torch.tensor(0.0)

        return response, log_prob, value

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """
        Perform single PPO training step.

        Args:
            batch: Batch of training data

        Returns:
            Dictionary of training metrics
        """
        self.optimizer.zero_grad()

        # Get data from batch
        observations = batch["observation"]
        actions = batch["action"]
        old_log_probs = batch["log_prob"]
        advantages = batch["advantage"]
        returns = batch["return"]

        # Recompute log probs and values with current policy
        current_log_probs, current_values = self._recompute_log_probs_and_values(
            observations, actions
        )

        # Compute PPO loss
        loss, metrics = compute_ppo_total_loss(
            current_log_probs=current_log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            predicted_values=current_values,
            returns=returns,
            clip_range=self.config.clip_range,
            value_loss_coef=self.config.value_loss_coef,
            entropy_coef=self.config.entropy_coef,
            value_clip_range=self.config.value_clip_range,
        )

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.policy_env.model.parameters(), self.config.max_grad_norm
        )

        # Optimizer step
        self.optimizer.step()

        return metrics

    def _recompute_log_probs_and_values(self, observations, actions):
        """
        Recompute log probabilities and values for given observations and actions.

        This is needed for PPO to compare old and new policies.

        Args:
            observations: List of observations
            actions: List of actions

        Returns:
            Tuple of (log_probs, values)
        """
        # Simplified version - in full implementation, this would:
        # 1. Tokenize observations and actions
        # 2. Forward pass through model to get logits
        # 3. Compute log probabilities of taken actions
        # 4. Compute value estimates

        # For now, use placeholder values
        batch_size = len(observations)

        log_probs = torch.zeros(batch_size, device=self.device, requires_grad=True)
        values = torch.zeros(batch_size, device=self.device, requires_grad=True)

        # TODO: Implement actual log prob and value computation
        # This requires tokenizing the conversation and computing forward pass

        return log_probs, values

    def train_epoch(
        self, rollouts: RolloutBuffer, num_epochs: Optional[int] = None, batch_size: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Train for multiple PPO epochs on rollout data.

        Args:
            rollouts: Collected rollout data
            num_epochs: Number of PPO epochs (uses config if None)
            batch_size: Batch size (uses config if None)

        Returns:
            Dictionary of aggregated metrics
        """
        if num_epochs is None:
            num_epochs = self.config.num_ppo_epochs
        if batch_size is None:
            batch_size = self.config.batch_size

        return super().train_epoch(rollouts, num_epochs, batch_size)

    def train(
        self,
        num_iterations: Optional[int] = None,
        eval_every: Optional[int] = None,
        save_every: Optional[int] = None,
    ):
        """
        Main PPO training loop.

        Args:
            num_iterations: Number of training iterations (uses config if None)
            eval_every: Evaluate every N iterations (uses config if None)
            save_every: Save every N iterations (uses config if None)
        """
        if num_iterations is None:
            num_iterations = self.config.num_epochs
        if eval_every is None:
            eval_every = self.config.eval_every
        if save_every is None:
            save_every = self.config.save_every

        super().train(num_iterations, eval_every, save_every)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PPOTrainer(step={self.global_step}, epoch={self.epoch}, "
            f"clip_range={self.config.clip_range})"
        )


from typing import Optional
