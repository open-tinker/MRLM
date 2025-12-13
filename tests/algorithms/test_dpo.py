"""Tests for DPO algorithm."""

import pytest
import torch
from mrlm.algorithms.dpo.loss import compute_dpo_loss
from mrlm.algorithms.dpo.dataset import PreferenceDataset, PreferencePair


class TestDPOLoss:
    """Test DPO loss functions."""

    def test_dpo_loss_computation(self):
        """Test DPO loss computation."""
        batch_size = 4

        # Log probabilities for chosen and rejected responses
        policy_chosen_logps = torch.randn(batch_size)
        policy_rejected_logps = torch.randn(batch_size)
        reference_chosen_logps = torch.randn(batch_size)
        reference_rejected_logps = torch.randn(batch_size)

        loss = compute_dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=reference_chosen_logps,
            reference_rejected_logps=reference_rejected_logps,
            beta=0.1,
        )

        assert isinstance(loss.item(), float)
        assert not torch.isnan(loss)
        assert loss >= 0  # DPO loss should be non-negative

    def test_dpo_loss_gradient_flow(self):
        """Test gradient flow through DPO loss."""
        batch_size = 4

        policy_chosen_logps = torch.randn(batch_size, requires_grad=True)
        policy_rejected_logps = torch.randn(batch_size, requires_grad=True)
        reference_chosen_logps = torch.randn(batch_size)
        reference_rejected_logps = torch.randn(batch_size)

        loss = compute_dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            beta=0.1,
        )

        loss.backward()

        assert policy_chosen_logps.grad is not None
        assert policy_rejected_logps.grad is not None

    def test_dpo_loss_beta_effect(self):
        """Test that beta parameter affects loss magnitude."""
        batch_size = 4

        policy_chosen_logps = torch.randn(batch_size)
        policy_rejected_logps = torch.randn(batch_size)
        reference_chosen_logps = torch.randn(batch_size)
        reference_rejected_logps = torch.randn(batch_size)

        # Compute loss with different beta values
        loss_beta_01 = compute_dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            beta=0.1,
        )

        loss_beta_10 = compute_dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            beta=1.0,
        )

        # Different beta should give different loss
        assert not torch.isclose(loss_beta_01, loss_beta_10)


class TestPreferenceDataset:
    """Test PreferenceDataset class."""

    def test_preference_pair_creation(self):
        """Test creating a preference pair."""
        pair = PreferencePair(
            prompt="What is 2+2?",
            chosen="2+2 equals 4",
            rejected="2+2 equals 5",
        )

        assert pair.prompt == "What is 2+2?"
        assert pair.chosen == "2+2 equals 4"
        assert pair.rejected == "2+2 equals 5"

    def test_preference_dataset_creation(self):
        """Test creating a preference dataset."""
        pairs = [
            PreferencePair("Prompt 1", "Good response", "Bad response"),
            PreferencePair("Prompt 2", "Better answer", "Worse answer"),
        ]

        dataset = PreferenceDataset(pairs)
        assert len(dataset) == 2

    def test_preference_dataset_indexing(self):
        """Test indexing into dataset."""
        pairs = [
            PreferencePair("Prompt 1", "Good", "Bad"),
            PreferencePair("Prompt 2", "Better", "Worse"),
        ]

        dataset = PreferenceDataset(pairs)

        pair = dataset[0]
        assert pair.prompt == "Prompt 1"
        assert pair.chosen == "Good"

    def test_preference_dataset_iteration(self):
        """Test iterating over dataset."""
        pairs = [
            PreferencePair(f"Prompt {i}", f"Chosen {i}", f"Rejected {i}")
            for i in range(5)
        ]

        dataset = PreferenceDataset(pairs)

        count = 0
        for pair in dataset:
            assert isinstance(pair, PreferencePair)
            count += 1

        assert count == 5

    def test_preference_dataset_save_load(self, temp_dir):
        """Test saving and loading dataset."""
        pairs = [
            PreferencePair("Prompt 1", "Good", "Bad"),
            PreferencePair("Prompt 2", "Better", "Worse"),
        ]

        dataset = PreferenceDataset(pairs)

        # Save
        save_path = temp_dir / "preferences.json"
        dataset.save(save_path)

        assert save_path.exists()

        # Load
        loaded_dataset = PreferenceDataset.load(save_path)

        assert len(loaded_dataset) == len(dataset)
        assert loaded_dataset[0].prompt == dataset[0].prompt
        assert loaded_dataset[1].chosen == dataset[1].chosen


@pytest.mark.slow
class TestDPOTrainer:
    """Test DPO trainer."""

    def test_dpo_trainer_creation(self, model, tokenizer, temp_dir):
        """Test creating DPO trainer."""
        from mrlm.algorithms.dpo import DPOTrainer
        from mrlm.core.llm_environment import LLMEnvironment
        from mrlm.core.types import EnvironmentMode
        from mrlm.config import ExperimentConfig, TrainingConfig, DPOConfig

        # Create preference dataset
        pairs = [
            PreferencePair("What is 2+2?", "4", "5"),
            PreferencePair("What is 3+3?", "6", "7"),
        ]
        dataset = PreferenceDataset(pairs)

        # Create policy environment
        policy_env = LLMEnvironment(model, tokenizer, mode=EnvironmentMode.SERVER)

        # Create config
        config = ExperimentConfig(
            training=TrainingConfig(
                algorithm="dpo",
                num_epochs=1,
                batch_size=2,
            ),
            dpo=DPOConfig(beta=0.1),
        )

        # Create trainer
        trainer = DPOTrainer(
            policy_env=policy_env,
            preference_dataset=dataset,
            config=config,
            device=torch.device("cpu"),
        )

        assert trainer is not None
        assert trainer.config.dpo.beta == 0.1
