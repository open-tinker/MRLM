"""Tests for core types."""

import pytest
from mrlm.core.types import (
    Message,
    MessageRole,
    Observation,
    Reward,
    RolloutBatch,
    EnvironmentMode,
)


class TestMessage:
    """Test Message class."""

    def test_message_creation(self):
        """Test creating a message."""
        msg = Message(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.metadata == {}

    def test_message_with_metadata(self):
        """Test message with metadata."""
        msg = Message(
            role=MessageRole.ASSISTANT,
            content="Hi there",
            metadata={"temperature": 0.7},
        )
        assert msg.metadata["temperature"] == 0.7

    def test_message_role_string(self):
        """Test message role as string."""
        msg = Message(role="user", content="Test")
        assert msg.role == "user"

    def test_message_to_dict(self):
        """Test converting message to dict."""
        msg = Message(role=MessageRole.USER, content="Hello")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "Hello"

    def test_message_from_dict(self):
        """Test creating message from dict."""
        d = {"role": "assistant", "content": "Hi"}
        msg = Message.from_dict(d)
        assert msg.role == "assistant"
        assert msg.content == "Hi"


class TestObservation:
    """Test Observation class."""

    def test_observation_creation(self, sample_messages):
        """Test creating an observation."""
        obs = Observation(
            messages=sample_messages,
            state={"step": 1},
            done=False,
            info={"test": True},
        )
        assert len(obs.messages) == 3
        assert obs.state["step"] == 1
        assert not obs.done
        assert obs.info["test"]

    def test_observation_minimal(self):
        """Test creating minimal observation."""
        obs = Observation(messages=[], done=False)
        assert obs.messages == []
        assert obs.state is None
        assert obs.info == {}

    def test_observation_to_dict(self, sample_observation):
        """Test converting observation to dict."""
        d = sample_observation.to_dict()
        assert "messages" in d
        assert "done" in d
        assert len(d["messages"]) == 3

    def test_observation_from_dict(self):
        """Test creating observation from dict."""
        d = {
            "messages": [{"role": "user", "content": "Test"}],
            "done": False,
            "state": {"step": 1},
            "info": {},
        }
        obs = Observation.from_dict(d)
        assert len(obs.messages) == 1
        assert not obs.done


class TestReward:
    """Test Reward class."""

    def test_reward_creation(self):
        """Test creating a reward."""
        reward = Reward(value=1.0, done=False, info={"success": True})
        assert reward.value == 1.0
        assert not reward.done
        assert reward.info["success"]

    def test_reward_terminal(self):
        """Test terminal reward."""
        reward = Reward(value=10.0, done=True)
        assert reward.done
        assert reward.value == 10.0

    def test_reward_zero(self):
        """Test zero reward."""
        reward = Reward(value=0.0, done=False)
        assert reward.value == 0.0

    def test_reward_negative(self):
        """Test negative reward."""
        reward = Reward(value=-5.0, done=False)
        assert reward.value == -5.0

    def test_reward_to_dict(self, sample_reward):
        """Test converting reward to dict."""
        d = sample_reward.to_dict()
        assert d["value"] == 1.0
        assert not d["done"]

    def test_reward_from_dict(self):
        """Test creating reward from dict."""
        d = {"value": 5.0, "done": True, "info": {}}
        reward = Reward.from_dict(d)
        assert reward.value == 5.0
        assert reward.done


class TestRolloutBatch:
    """Test RolloutBatch class."""

    def test_rollout_batch_creation(self):
        """Test creating a rollout batch."""
        import torch

        batch = RolloutBatch(
            observations=[],
            actions=[],
            rewards=torch.tensor([1.0, 2.0, 3.0]),
            values=torch.tensor([0.5, 1.0, 1.5]),
            log_probs=torch.tensor([-0.1, -0.2, -0.3]),
            dones=torch.tensor([0, 0, 1], dtype=torch.bool),
        )

        assert len(batch.rewards) == 3
        assert len(batch.values) == 3
        assert batch.dones[-1] == True

    def test_rollout_batch_advantages(self):
        """Test computing advantages."""
        import torch

        batch = RolloutBatch(
            observations=[],
            actions=[],
            rewards=torch.tensor([1.0, 2.0, 3.0]),
            values=torch.tensor([0.5, 1.0, 1.5]),
            log_probs=torch.tensor([-0.1, -0.2, -0.3]),
            dones=torch.tensor([0, 0, 1], dtype=torch.bool),
            advantages=torch.tensor([0.5, 1.0, 1.5]),
            returns=torch.tensor([1.5, 3.0, 4.5]),
        )

        assert batch.advantages is not None
        assert len(batch.advantages) == 3


class TestEnvironmentMode:
    """Test EnvironmentMode enum."""

    def test_server_mode(self):
        """Test SERVER mode."""
        assert EnvironmentMode.SERVER.value == "server"

    def test_client_mode(self):
        """Test CLIENT mode."""
        assert EnvironmentMode.CLIENT.value == "client"

    def test_mode_equality(self):
        """Test mode equality."""
        assert EnvironmentMode.SERVER == EnvironmentMode.SERVER
        assert EnvironmentMode.CLIENT != EnvironmentMode.SERVER
