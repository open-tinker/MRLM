"""Tests for core base classes."""

import pytest
from mrlm.core.base import BaseEnvironment, SimulatedEnvironment
from mrlm.core.types import Message, Observation, Reward, EnvironmentMode, MessageRole


class DummySimulatedEnvironment(SimulatedEnvironment):
    """Dummy simulated environment for testing."""

    def __init__(self, mode=EnvironmentMode.CLIENT, max_turns=3):
        super().__init__(mode=mode, max_turns=max_turns)
        self.reset_count = 0
        self.step_count = 0

    def reset(self) -> Observation:
        """Reset the environment."""
        self.reset_count += 1
        self.current_turn = 0
        return Observation(
            messages=[Message(role=MessageRole.SYSTEM, content="Reset")],
            done=False,
        )

    def step(self, action: Message) -> tuple[Observation, Reward]:
        """Take a step in the environment."""
        self.step_count += 1
        self.current_turn += 1

        done = self.current_turn >= self.max_turns
        reward_value = 1.0 if "correct" in action.content.lower() else 0.0

        obs = Observation(
            messages=[action],
            done=done,
            state={"turn": self.current_turn},
        )
        reward = Reward(value=reward_value, done=done)

        return obs, reward


class TestBaseEnvironment:
    """Test BaseEnvironment abstract class."""

    def test_environment_creation(self):
        """Test creating an environment."""
        env = DummySimulatedEnvironment(mode=EnvironmentMode.CLIENT)
        assert env.mode == EnvironmentMode.CLIENT

    def test_environment_reset(self):
        """Test resetting environment."""
        env = DummySimulatedEnvironment()
        obs = env.reset()
        assert env.reset_count == 1
        assert len(obs.messages) == 1
        assert not obs.done

    def test_environment_step(self):
        """Test stepping in environment."""
        env = DummySimulatedEnvironment()
        env.reset()

        action = Message(role=MessageRole.ASSISTANT, content="This is correct")
        obs, reward = env.step(action)

        assert env.step_count == 1
        assert reward.value == 1.0
        assert not obs.done

    def test_environment_episode_length(self):
        """Test environment respects max_turns."""
        env = DummySimulatedEnvironment(max_turns=2)
        env.reset()

        # First step
        action1 = Message(role=MessageRole.ASSISTANT, content="Step 1")
        obs1, reward1 = env.step(action1)
        assert not obs1.done

        # Second step (should terminate)
        action2 = Message(role=MessageRole.ASSISTANT, content="Step 2")
        obs2, reward2 = env.step(action2)
        assert obs2.done

    def test_environment_close(self):
        """Test closing environment."""
        env = DummySimulatedEnvironment()
        env.reset()
        env.close()  # Should not raise


class TestSimulatedEnvironment:
    """Test SimulatedEnvironment base class."""

    def test_simulated_env_turn_tracking(self):
        """Test turn tracking in simulated environment."""
        env = DummySimulatedEnvironment(max_turns=5)
        env.reset()

        for i in range(3):
            action = Message(role=MessageRole.ASSISTANT, content=f"Turn {i}")
            obs, reward = env.step(action)
            assert env.current_turn == i + 1

    def test_simulated_env_state(self):
        """Test state tracking."""
        env = DummySimulatedEnvironment()
        obs = env.reset()
        assert env.current_turn == 0

        action = Message(role=MessageRole.ASSISTANT, content="Test")
        obs, reward = env.step(action)
        assert obs.state["turn"] == 1

    def test_multiple_episodes(self):
        """Test running multiple episodes."""
        env = DummySimulatedEnvironment(max_turns=2)

        # Episode 1
        env.reset()
        assert env.reset_count == 1

        action = Message(role=MessageRole.ASSISTANT, content="correct")
        env.step(action)
        env.step(action)

        # Episode 2
        env.reset()
        assert env.reset_count == 2
        assert env.current_turn == 0

        env.step(action)
        env.step(action)

        assert env.step_count == 4
