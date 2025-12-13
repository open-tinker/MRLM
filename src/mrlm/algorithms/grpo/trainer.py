"""
GRPO (Group Relative Policy Optimization) trainer.

GRPO trainer for language model fine-tuning with group-normalized rewards.
"""

from typing import Dict, List, Optional, Any
import random

import torch
import torch.nn as nn
from torch.optim import Optimizer, AdamW
from transformers import PreTrainedModel, PreTrainedTokenizer

from mrlm.core.base import BaseEnvironment
from mrlm.core.types import Message, Observation, Reward
from mrlm.data.buffer import RolloutBuffer
from mrlm.algorithms.base_trainer import BaseTrainer
from mrlm.algorithms.grpo.loss import (
    compute_grpo_loss,
    compute_group_advantages,
    compute_value_loss,
    compute_entropy_bonus,
    normalize_rewards_by_group,
)
from mrlm.config.training_config import ExperimentConfig


class GRPOTrainer(BaseTrainer):
    """
    GRPO trainer for language model fine-tuning.

    GRPO uses group-normalized rewards to reduce variance in policy gradient
    estimates. This is particularly useful for LLM training where different
    prompts may have different reward scales.
    """

    def __init__(
        self,
        policy_env: BaseEnvironment,
        eval_envs: List[BaseEnvironment],
        config: ExperimentConfig,
        optimizer: Optional[Optimizer] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize GRPO trainer.

        Args:
            policy_env: LLM environment in SERVER mode (for training)
            eval_envs: List of evaluation environments
            config: Experiment configuration
            optimizer: Optional optimizer (created if not provided)
            device: Device for training
        """
        super().__init__(policy_env, eval_envs, config, device)

        # GRPO-specific config
        self.grpo_config = config.grpo
        if self.grpo_config is None:
            raise ValueError("GRPO config must be provided for GRPOTrainer")

        # Get model and create optimizer if needed
        if hasattr(policy_env, "model"):
            self.model: PreTrainedModel = policy_env.model
            if optimizer is None:
                self.optimizer = AdamW(
                    self.model.parameters(),
                    lr=config.training.learning_rate,
                    weight_decay=config.training.weight_decay,
                )
            else:
                self.optimizer = optimizer
        else:
            raise ValueError("Policy environment must have a 'model' attribute")

        # Group size for GRPO (number of responses per prompt)
        self.group_size = self.grpo_config.group_size

    def collect_rollouts(self) -> RolloutBuffer:
        """
        Collect rollouts using the current policy.

        For GRPO, we generate multiple responses per prompt to form groups.

        Returns:
            RolloutBuffer with collected trajectories and group IDs
        """
        buffer = RolloutBuffer()
        total_episodes = 0
        target_episodes = self.config.training.episodes_per_iteration

        self.model.eval()
        with torch.no_grad():
            while total_episodes < target_episodes:
                # Randomly select an environment
                env = random.choice(self.eval_envs)
                obs = env.reset()

                # For GRPO, we generate multiple completions for the same prompt
                # Store the initial observation for the group
                group_id = total_episodes // self.group_size

                for response_idx in range(self.group_size):
                    if total_episodes >= target_episodes:
                        break

                    # Reset to same initial state for group members
                    if response_idx > 0:
                        obs = env.reset()

                    episode_length = 0
                    done = False

                    while not done and episode_length < self.config.training.max_episode_length:
                        # Generate action with log probabilities and value
                        action, log_prob, value = self._generate_action_with_value(
                            obs.messages
                        )

                        # Take step in environment
                        next_obs, reward = env.step(action)
                        done = next_obs.done

                        # Store transition with group ID
                        buffer.add(
                            observation=obs,
                            action=action,
                            reward=reward,
                            log_prob=log_prob,
                            value=value,
                            done=done,
                            info={"group_id": group_id},
                        )

                        obs = next_obs
                        episode_length += 1

                        if done:
                            break

                    total_episodes += 1

        # Compute group-normalized advantages
        self._compute_grpo_advantages(buffer)

        return buffer

    def _compute_grpo_advantages(self, buffer: RolloutBuffer):
        """
        Compute advantages using group normalization.

        Args:
            buffer: Rollout buffer to process
        """
        if len(buffer) == 0:
            return

        # Extract data
        rewards = torch.tensor([r.value for r in buffer.rewards], dtype=torch.float32)
        values = torch.stack(buffer.values) if buffer.values else torch.zeros_like(rewards)
        group_ids = torch.tensor(
            [info.get("group_id", 0) for info in buffer.info],
            dtype=torch.long,
        )

        # Compute group-normalized advantages
        advantages = compute_group_advantages(
            rewards=rewards,
            values=values,
            group_ids=group_ids,
            gamma=self.grpo_config.gamma,
            normalize=True,
        )

        # Store advantages and returns in buffer
        buffer.advantages = advantages
        buffer.returns = advantages + values  # For GRPO, returns = advantages + baseline

    def train_epoch(self, rollouts: RolloutBuffer) -> Dict[str, float]:
        """
        Train for one epoch on collected rollouts.

        Args:
            rollouts: Buffer containing rollout data

        Returns:
            Dictionary of training metrics
        """
        if len(rollouts) == 0:
            return {}

        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "total_loss": 0.0,
            "clip_fraction": 0.0,
            "approx_kl": 0.0,
        }

        # Convert to tensors
        old_log_probs = torch.stack(rollouts.log_probs).to(self.device)
        advantages = rollouts.advantages.to(self.device)
        returns = rollouts.returns.to(self.device)
        old_values = torch.stack(rollouts.values).to(self.device) if rollouts.values else None
        group_ids = torch.tensor(
            [info.get("group_id", 0) for info in rollouts.info],
            dtype=torch.long,
        ).to(self.device)

        # Train for multiple epochs
        num_updates = 0
        for epoch in range(self.grpo_config.num_grpo_epochs):
            # Create mini-batches
            indices = torch.randperm(len(rollouts))
            batch_size = self.config.training.batch_size

            for start_idx in range(0, len(rollouts), batch_size):
                end_idx = min(start_idx + batch_size, len(rollouts))
                batch_indices = indices[start_idx:end_idx]

                # Get batch data
                batch_obs = [rollouts.observations[i] for i in batch_indices]
                batch_actions = [rollouts.actions[i] for i in batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_group_ids = group_ids[batch_indices]

                # Recompute log probs and values with current policy
                self.model.train()
                current_log_probs_list = []
                current_values_list = []

                for obs, action in zip(batch_obs, batch_actions):
                    _, log_prob, value = self._generate_action_with_value(
                        obs.messages, forced_action=action
                    )
                    current_log_probs_list.append(log_prob)
                    current_values_list.append(value)

                current_log_probs = torch.stack(current_log_probs_list)
                current_values = torch.stack(current_values_list)

                # Compute losses
                policy_loss, policy_info = compute_grpo_loss(
                    current_log_probs=current_log_probs,
                    old_log_probs=batch_old_log_probs,
                    advantages=batch_advantages,
                    clip_range=self.grpo_config.clip_range,
                    group_ids=batch_group_ids,
                )

                value_loss, value_info = compute_value_loss(
                    values=current_values,
                    returns=batch_returns,
                    clip_range=self.grpo_config.value_clip_range,
                )

                entropy_loss, entropy_info = compute_entropy_bonus(
                    log_probs=current_log_probs,
                    entropy_coef=self.grpo_config.entropy_coef,
                )

                # Total loss
                total_loss = (
                    policy_loss
                    + self.grpo_config.value_loss_coef * value_loss
                    + entropy_loss
                )

                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()

                # Clip gradients
                if self.config.training.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.max_grad_norm,
                    )

                self.optimizer.step()

                # Accumulate metrics
                metrics["policy_loss"] += policy_info["policy_loss"]
                metrics["value_loss"] += value_info["value_loss"]
                metrics["entropy_loss"] += entropy_info["entropy_loss"]
                metrics["total_loss"] += total_loss.item()
                metrics["clip_fraction"] += policy_info["clip_fraction"]
                metrics["approx_kl"] += policy_info["approx_kl"]

                num_updates += 1

                # Early stopping if KL divergence is too high
                if (
                    self.grpo_config.target_kl is not None
                    and policy_info["approx_kl"] > self.grpo_config.target_kl
                ):
                    print(
                        f"Early stopping at epoch {epoch} due to high KL: {policy_info['approx_kl']:.4f}"
                    )
                    break

        # Average metrics
        if num_updates > 0:
            for key in metrics:
                metrics[key] /= num_updates

        return metrics

    def _generate_action_with_value(
        self,
        messages: List[Message],
        forced_action: Optional[Message] = None,
    ) -> tuple[Message, torch.Tensor, torch.Tensor]:
        """
        Generate action with log probability and value estimate.

        Args:
            messages: Conversation history
            forced_action: If provided, compute log prob for this action

        Returns:
            Tuple of (action_message, log_prob, value)
        """
        # Use policy_env to generate action
        # For forced action, we need to compute its log probability
        if forced_action is not None:
            # This is during training - recompute log prob for forced action
            from mrlm.models.generation import compute_log_probs

            text, log_prob, value = compute_log_probs(
                model=self.model,
                tokenizer=self.policy_env.tokenizer,
                messages=messages,
                response_text=forced_action.content,
                device=self.device,
                return_value=True,
            )
            return forced_action, log_prob, value
        else:
            # This is during rollout collection - generate new action
            from mrlm.models.generation import generate_with_log_probs

            text, log_prob, value = generate_with_log_probs(
                model=self.model,
                tokenizer=self.policy_env.tokenizer,
                messages=messages,
                generation_config=self.policy_env.generation_config,
                device=self.device,
                return_value=True,
            )

            action = Message(role="assistant", content=text)
            return action, log_prob, value

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
            for env in self.eval_envs:
                obs = env.reset()
                episode_reward = 0.0
                episode_length = 0
                done = False

                while not done and episode_length < self.config.training.max_episode_length:
                    # Generate action (no value needed for eval)
                    action, _, _ = self._generate_action_with_value(obs.messages)
                    next_obs, reward = env.step(action)

                    episode_reward += reward.value
                    episode_length += 1
                    done = next_obs.done
                    obs = next_obs

                total_reward += episode_reward
                num_episodes += 1

        avg_reward = total_reward / max(num_episodes, 1)
        return {"eval/avg_reward": avg_reward, "eval/num_episodes": num_episodes}
