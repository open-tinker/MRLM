"""Tests for GRPO algorithm."""

import pytest
import torch
from mrlm.algorithms.grpo.loss import compute_grpo_loss, normalize_rewards_in_groups


class TestGRPOLoss:
    """Test GRPO loss functions."""

    def test_normalize_rewards_in_groups(self):
        """Test group-wise reward normalization."""
        # 2 groups of 4 samples each
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
        group_size = 4

        normalized = normalize_rewards_in_groups(rewards, group_size)

        assert normalized.shape == rewards.shape

        # Check first group normalization
        group1 = normalized[:4]
        assert torch.abs(group1.mean()) < 0.1

        # Check second group normalization
        group2 = normalized[4:]
        assert torch.abs(group2.mean()) < 0.1

    def test_normalize_rewards_single_group(self):
        """Test normalization with single group."""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        group_size = 4

        normalized = normalize_rewards_in_groups(rewards, group_size)

        # Should be normalized to mean 0
        assert torch.abs(normalized.mean()) < 0.1

    def test_normalize_rewards_multiple_groups(self):
        """Test with multiple groups."""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        group_size = 3

        normalized = normalize_rewards_in_groups(rewards, group_size)

        # Each group should be normalized separately
        assert normalized.shape == rewards.shape
        assert torch.abs(normalized[:3].mean()) < 0.1
        assert torch.abs(normalized[3:].mean()) < 0.1

    def test_grpo_loss_computation(self):
        """Test GRPO loss computation."""
        batch_size = 8
        group_size = 4

        log_probs = torch.randn(batch_size)
        normalized_rewards = torch.randn(batch_size)

        loss = compute_grpo_loss(
            log_probs=log_probs,
            normalized_rewards=normalized_rewards,
            group_size=group_size,
        )

        assert isinstance(loss.item(), float)
        assert not torch.isnan(loss)

    def test_grpo_loss_gradient_flow(self):
        """Test that GRPO loss allows gradient flow."""
        batch_size = 8
        group_size = 4

        log_probs = torch.randn(batch_size, requires_grad=True)
        normalized_rewards = torch.randn(batch_size)

        loss = compute_grpo_loss(log_probs, normalized_rewards, group_size)
        loss.backward()

        assert log_probs.grad is not None
        assert not torch.isnan(log_probs.grad).any()


@pytest.mark.slow
class TestGRPOTrainer:
    """Test GRPO trainer."""

    def test_grpo_trainer_creation(self, model, tokenizer):
        """Test creating GRPO trainer."""
        from mrlm.algorithms.grpo import GRPOTrainer
        from mrlm.core.llm_environment import LLMEnvironment
        from mrlm.core.types import EnvironmentMode
        from mrlm.config import ExperimentConfig, TrainingConfig, GRPOConfig
        from tests.core.test_base import DummySimulatedEnvironment

        policy_env = LLMEnvironment(model, tokenizer, mode=EnvironmentMode.SERVER)
        eval_envs = [DummySimulatedEnvironment() for _ in range(2)]

        config = ExperimentConfig(
            training=TrainingConfig(
                algorithm="grpo",
                num_epochs=1,
                batch_size=4,
                max_episode_length=2,
                episodes_per_iteration=4,
            ),
            grpo=GRPOConfig(
                group_size=4,
                clip_range=0.2,
                gamma=0.99,
            ),
        )

        trainer = GRPOTrainer(
            policy_env=policy_env,
            eval_envs=eval_envs,
            config=config,
            device=torch.device("cpu"),
        )

        assert trainer is not None
        assert trainer.config.grpo.group_size == 4

    def test_grpo_group_size_validation(self):
        """Test that group size divides batch size."""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])  # Length 5
        group_size = 4  # Doesn't divide evenly

        # Should handle this gracefully or raise informative error
        try:
            normalized = normalize_rewards_in_groups(rewards, group_size)
            # If it works, check the result
            assert normalized.shape == rewards.shape
        except ValueError as e:
            # If it raises an error, it should be informative
            assert "group_size" in str(e).lower() or "batch" in str(e).lower()
