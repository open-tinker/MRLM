"""Tests for SFT algorithm."""

import pytest
import torch
from mrlm.algorithms.sft.dataset import Trajectory, TrajectoryDataset
from mrlm.algorithms.sft.loss import (
    compute_behavioral_cloning_loss,
    compute_next_state_loss,
    compute_combined_sft_loss,
)
from mrlm.core.types import Message, Observation, Reward, MessageRole


class TestTrajectory:
    """Test Trajectory class."""

    def test_trajectory_creation(self, sample_observation, sample_reward):
        """Test creating a trajectory."""
        traj = Trajectory(
            observations=[sample_observation],
            actions=[Message(role=MessageRole.ASSISTANT, content="Action 1")],
            rewards=[sample_reward],
        )

        assert len(traj.observations) == 1
        assert len(traj.actions) == 1
        assert len(traj.rewards) == 1

    def test_trajectory_total_reward(self):
        """Test computing total reward."""
        rewards = [
            Reward(value=1.0, done=False),
            Reward(value=2.0, done=False),
            Reward(value=3.0, done=True),
        ]

        traj = Trajectory(observations=[], actions=[], rewards=rewards)
        total = traj.total_reward()

        assert total == 6.0

    def test_trajectory_length(self):
        """Test trajectory length."""
        traj = Trajectory(
            observations=[Observation(messages=[], done=False) for _ in range(5)],
            actions=[
                Message(role=MessageRole.ASSISTANT, content=f"A{i}") for i in range(5)
            ],
            rewards=[Reward(value=1.0, done=False) for _ in range(5)],
        )

        assert len(traj) == 5

    def test_trajectory_to_dict(self):
        """Test converting trajectory to dict."""
        traj = Trajectory(
            observations=[Observation(messages=[], done=False)],
            actions=[Message(role=MessageRole.ASSISTANT, content="Test")],
            rewards=[Reward(value=1.0, done=False)],
            metadata={"episode": 1},
        )

        d = traj.to_dict()

        assert "observations" in d
        assert "actions" in d
        assert "rewards" in d
        assert "metadata" in d

    def test_trajectory_from_dict(self):
        """Test creating trajectory from dict."""
        d = {
            "observations": [{"messages": [], "done": False, "state": None, "info": {}}],
            "actions": [{"role": "assistant", "content": "Test", "metadata": {}}],
            "rewards": [{"value": 1.0, "done": False, "info": {}}],
            "metadata": {},
        }

        traj = Trajectory.from_dict(d)

        assert len(traj.observations) == 1
        assert len(traj.actions) == 1
        assert len(traj.rewards) == 1


class TestTrajectoryDataset:
    """Test TrajectoryDataset class."""

    def test_dataset_creation(self):
        """Test creating a trajectory dataset."""
        trajectories = [
            Trajectory(
                observations=[Observation(messages=[], done=False)],
                actions=[Message(role=MessageRole.ASSISTANT, content="A1")],
                rewards=[Reward(value=1.0, done=False)],
            )
        ]

        dataset = TrajectoryDataset(trajectories)
        assert len(dataset) == 1

    def test_dataset_add_trajectory(self):
        """Test adding trajectories to dataset."""
        dataset = TrajectoryDataset()

        traj = Trajectory(
            observations=[Observation(messages=[], done=False)],
            actions=[Message(role=MessageRole.ASSISTANT, content="Test")],
            rewards=[Reward(value=1.0, done=False)],
        )

        dataset.add_trajectory(traj)
        assert len(dataset) == 1

    def test_dataset_filter_by_reward(self):
        """Test filtering trajectories by reward."""
        trajectories = [
            Trajectory(
                observations=[],
                actions=[],
                rewards=[Reward(value=0.5, done=False)],  # Low reward
            ),
            Trajectory(
                observations=[],
                actions=[],
                rewards=[Reward(value=2.0, done=False)],  # High reward
            ),
            Trajectory(
                observations=[],
                actions=[],
                rewards=[Reward(value=1.5, done=False)],  # Medium reward
            ),
        ]

        dataset = TrajectoryDataset(trajectories)

        # Filter for rewards >= 1.0
        filtered = dataset.filter_by_reward(min_reward=1.0)

        assert len(filtered) == 2  # Should keep 2.0 and 1.5
        assert filtered.trajectories[0].total_reward() >= 1.0

    def test_dataset_indexing(self):
        """Test indexing into dataset."""
        trajectories = [
            Trajectory(
                observations=[],
                actions=[],
                rewards=[Reward(value=float(i), done=False)],
            )
            for i in range(5)
        ]

        dataset = TrajectoryDataset(trajectories)

        assert dataset[0].total_reward() == 0.0
        assert dataset[2].total_reward() == 2.0

    def test_dataset_save_load(self, temp_dir):
        """Test saving and loading dataset."""
        trajectories = [
            Trajectory(
                observations=[Observation(messages=[], done=False)],
                actions=[Message(role=MessageRole.ASSISTANT, content="Test")],
                rewards=[Reward(value=1.0, done=False)],
            )
        ]

        dataset = TrajectoryDataset(trajectories)

        # Save
        save_path = temp_dir / "trajectories.json"
        dataset.save(save_path)

        assert save_path.exists()

        # Load
        loaded = TrajectoryDataset.load(save_path)

        assert len(loaded) == len(dataset)
        assert loaded[0].total_reward() == dataset[0].total_reward()


class TestSFTLoss:
    """Test SFT loss functions."""

    @pytest.mark.slow
    def test_behavioral_cloning_loss(self, model, tokenizer):
        """Test behavioral cloning loss computation."""
        observations = [
            Observation(
                messages=[Message(role=MessageRole.USER, content="What is 2+2?")],
                done=False,
            )
        ]
        actions = [Message(role=MessageRole.ASSISTANT, content="4")]

        loss = compute_behavioral_cloning_loss(
            model=model,
            tokenizer=tokenizer,
            observations=observations,
            actions=actions,
        )

        assert isinstance(loss.item(), float)
        assert not torch.isnan(loss)
        assert loss >= 0

    @pytest.mark.slow
    def test_next_state_loss(self, model, tokenizer):
        """Test world model loss computation."""
        observations = [
            Observation(
                messages=[Message(role=MessageRole.USER, content="Question")],
                done=False,
            )
        ]
        actions = [Message(role=MessageRole.ASSISTANT, content="Answer")]
        next_observations = [
            Observation(
                messages=[Message(role=MessageRole.USER, content="Next question")],
                done=False,
            )
        ]

        loss = compute_next_state_loss(
            model=model,
            tokenizer=tokenizer,
            observations=observations,
            actions=actions,
            next_observations=next_observations,
        )

        assert isinstance(loss.item(), float)
        assert not torch.isnan(loss)
        assert loss >= 0

    @pytest.mark.slow
    def test_combined_sft_loss(self, model, tokenizer):
        """Test combined SFT loss."""
        observations = [
            Observation(
                messages=[Message(role=MessageRole.USER, content="Q")],
                done=False,
            )
        ]
        actions = [Message(role=MessageRole.ASSISTANT, content="A")]
        next_observations = [
            Observation(
                messages=[Message(role=MessageRole.USER, content="Q2")],
                done=False,
            )
        ]

        loss = compute_combined_sft_loss(
            model=model,
            tokenizer=tokenizer,
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            bc_weight=0.6,
            world_model_weight=0.4,
        )

        assert isinstance(loss.item(), float)
        assert not torch.isnan(loss)
        assert loss >= 0


@pytest.mark.slow
class TestSFTTrainer:
    """Test SFT trainer."""

    def test_sft_trainer_creation(self, model, tokenizer):
        """Test creating SFT trainer."""
        from mrlm.algorithms.sft import SFTTrainer
        from mrlm.core.llm_environment import LLMEnvironment
        from mrlm.core.types import EnvironmentMode
        from mrlm.config import ExperimentConfig, TrainingConfig, SFTConfig
        from tests.core.test_base import DummySimulatedEnvironment

        policy_env = LLMEnvironment(model, tokenizer, mode=EnvironmentMode.SERVER)
        eval_envs = [DummySimulatedEnvironment() for _ in range(2)]

        config = ExperimentConfig(
            training=TrainingConfig(
                algorithm="sft",
                num_epochs=1,
                batch_size=2,
                max_episode_length=2,
                episodes_per_iteration=2,
            ),
            sft=SFTConfig(
                mode="combined",
                bc_weight=0.6,
                world_model_weight=0.4,
            ),
        )

        trainer = SFTTrainer(
            policy_env=policy_env,
            eval_envs=eval_envs,
            config=config,
            device=torch.device("cpu"),
        )

        assert trainer is not None
        assert trainer.config.sft.mode == "combined"
