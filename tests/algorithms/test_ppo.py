"""Tests for PPO algorithm."""

import pytest
import torch
from mrlm.algorithms.ppo.loss import compute_ppo_loss, compute_gae
from mrlm.algorithms.ppo.utils import normalize_advantages
from mrlm.core.types import RolloutBatch


class TestPPOLoss:
    """Test PPO loss functions."""

    def test_compute_gae(self):
        """Test GAE computation."""
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([0.5, 1.0, 1.5])
        dones = torch.tensor([0, 0, 1], dtype=torch.bool)
        next_value = torch.tensor(0.0)

        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            next_value=next_value,
            gamma=0.99,
            gae_lambda=0.95,
        )

        assert advantages.shape == rewards.shape
        assert returns.shape == rewards.shape
        assert not torch.isnan(advantages).any()
        assert not torch.isnan(returns).any()

    def test_compute_gae_all_done(self):
        """Test GAE with all episodes done."""
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([0.5, 1.0, 1.5])
        dones = torch.tensor([1, 1, 1], dtype=torch.bool)
        next_value = torch.tensor(0.0)

        advantages, returns = compute_gae(
            rewards, values, dones, next_value, gamma=0.99, gae_lambda=0.95
        )

        assert advantages.shape == rewards.shape
        assert returns.shape == rewards.shape

    def test_normalize_advantages(self):
        """Test advantage normalization."""
        advantages = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = normalize_advantages(advantages)

        # Should have approximately mean 0 and std 1
        assert torch.abs(normalized.mean()) < 0.1
        assert torch.abs(normalized.std() - 1.0) < 0.1

    def test_normalize_advantages_constant(self):
        """Test normalizing constant advantages."""
        advantages = torch.ones(10) * 5.0
        normalized = normalize_advantages(advantages)

        # All values should be close to zero for constant input
        assert torch.abs(normalized).max() < 0.1

    def test_ppo_loss_computation(self):
        """Test PPO loss computation."""
        batch_size = 8
        vocab_size = 100

        # Create dummy batch
        old_log_probs = torch.randn(batch_size)
        new_log_probs = torch.randn(batch_size)
        advantages = torch.randn(batch_size)
        returns = torch.randn(batch_size)
        values = torch.randn(batch_size)

        # Compute loss
        policy_loss, value_loss, entropy = compute_ppo_loss(
            old_log_probs=old_log_probs,
            new_log_probs=new_log_probs,
            advantages=advantages,
            returns=returns,
            values=values,
            clip_range=0.2,
            value_loss_coef=0.5,
        )

        assert isinstance(policy_loss.item(), float)
        assert isinstance(value_loss.item(), float)
        assert not torch.isnan(policy_loss)
        assert not torch.isnan(value_loss)

    def test_ppo_loss_clipping(self):
        """Test PPO clipping behavior."""
        batch_size = 4

        # Old and new log probs with large difference (should be clipped)
        old_log_probs = torch.tensor([0.0, 0.0, 0.0, 0.0])
        new_log_probs = torch.tensor([5.0, 5.0, -5.0, -5.0])  # Large differences
        advantages = torch.tensor([1.0, -1.0, 1.0, -1.0])
        returns = torch.tensor([1.0, 1.0, 1.0, 1.0])
        values = torch.tensor([0.5, 0.5, 0.5, 0.5])

        policy_loss, _, _ = compute_ppo_loss(
            old_log_probs=old_log_probs,
            new_log_probs=new_log_probs,
            advantages=advantages,
            returns=returns,
            values=values,
            clip_range=0.2,
            value_loss_coef=0.5,
        )

        # Should have finite loss due to clipping
        assert torch.isfinite(policy_loss)


class TestPPOUtils:
    """Test PPO utility functions."""

    def test_advantage_normalization_shape(self):
        """Test that normalization preserves shape."""
        for shape in [(10,), (5, 3), (2, 4, 3)]:
            advantages = torch.randn(shape)
            normalized = normalize_advantages(advantages)
            assert normalized.shape == advantages.shape

    def test_gae_discount_factor(self):
        """Test GAE with different discount factors."""
        rewards = torch.ones(5)
        values = torch.zeros(5)
        dones = torch.zeros(5, dtype=torch.bool)
        next_value = torch.tensor(0.0)

        # With gamma=0, should only care about immediate rewards
        adv_0, ret_0 = compute_gae(
            rewards, values, dones, next_value, gamma=0.0, gae_lambda=0.95
        )

        # With gamma=0.99, should care about future rewards
        adv_99, ret_99 = compute_gae(
            rewards, values, dones, next_value, gamma=0.99, gae_lambda=0.95
        )

        # Returns with higher gamma should be larger
        assert ret_99.sum() > ret_0.sum()


@pytest.mark.slow
class TestPPOTrainer:
    """Test PPO trainer (requires model)."""

    def test_trainer_creation(self, model, tokenizer):
        """Test creating PPO trainer."""
        from mrlm.algorithms.ppo import PPOTrainer
        from mrlm.core.llm_environment import LLMEnvironment
        from mrlm.core.types import EnvironmentMode
        from mrlm.config import ExperimentConfig, TrainingConfig, PPOConfig

        # Create environments
        policy_env = LLMEnvironment(model, tokenizer, mode=EnvironmentMode.SERVER)

        # Create simple dummy eval env
        from tests.core.test_base import DummySimulatedEnvironment

        eval_envs = [DummySimulatedEnvironment() for _ in range(2)]

        # Create config
        config = ExperimentConfig(
            training=TrainingConfig(
                algorithm="ppo",
                num_epochs=1,
                batch_size=2,
                max_episode_length=2,
                episodes_per_iteration=2,
            ),
            ppo=PPOConfig(clip_range=0.2, gamma=0.99),
        )

        # Create trainer
        trainer = PPOTrainer(
            policy_env=policy_env,
            eval_envs=eval_envs,
            config=config,
            device=torch.device("cpu"),
        )

        assert trainer is not None
        assert trainer.config.training.algorithm == "ppo"

    @pytest.mark.requires_gpu
    def test_trainer_single_iteration(self, model, tokenizer):
        """Test running a single training iteration."""
        from mrlm.algorithms.ppo import PPOTrainer
        from mrlm.core.llm_environment import LLMEnvironment
        from mrlm.core.types import EnvironmentMode
        from mrlm.config import ExperimentConfig, TrainingConfig, PPOConfig
        from tests.core.test_base import DummySimulatedEnvironment

        policy_env = LLMEnvironment(model, tokenizer, mode=EnvironmentMode.SERVER)
        eval_envs = [DummySimulatedEnvironment(max_turns=2) for _ in range(2)]

        config = ExperimentConfig(
            training=TrainingConfig(
                algorithm="ppo",
                num_epochs=1,
                batch_size=2,
                max_episode_length=2,
                episodes_per_iteration=2,
            ),
            ppo=PPOConfig(
                clip_range=0.2,
                gamma=0.99,
                num_ppo_epochs=1,
            ),
        )

        trainer = PPOTrainer(policy_env, eval_envs, config, device=torch.device("cpu"))

        # Run one iteration
        trainer.train(num_iterations=1, eval_every=1)

        # Should have completed without errors
        assert True
