"""Tests for debate environment."""

import pytest
from mrlm.environments.debate import DebateEnvironment, RuleBasedJudge
from mrlm.core.types import Message, MessageRole, EnvironmentMode


class TestRuleBasedJudge:
    """Test RuleBasedJudge class."""

    def test_judge_evaluation(self):
        """Test evaluating debate arguments."""
        judge = RuleBasedJudge()

        topic = "AI is beneficial to society"
        pro_argument = "AI improves healthcare, education, and productivity."
        con_argument = "AI may replace jobs and raise privacy concerns."

        pro_score = judge.evaluate_argument(pro_argument, "PRO", topic)
        con_score = judge.evaluate_argument(con_argument, "CON", topic)

        assert 0 <= pro_score <= 1
        assert 0 <= con_score <= 1

    def test_judge_determine_winner(self):
        """Test determining debate winner."""
        judge = RuleBasedJudge()

        winner, pro_score, con_score = judge.determine_winner(
            pro_arguments=["Strong argument with evidence"],
            con_arguments=["Weak argument"],
        )

        assert winner in ["PRO", "CON", "TIE"]
        assert 0 <= pro_score <= 1
        assert 0 <= con_score <= 1


class TestDebateEnvironment:
    """Test DebateEnvironment class."""

    def test_environment_creation(self):
        """Test creating debate environment."""
        judge = RuleBasedJudge()
        env = DebateEnvironment(judge=judge, mode=EnvironmentMode.CLIENT)

        assert env.mode == EnvironmentMode.CLIENT
        assert env.judge is judge

    def test_environment_reset(self):
        """Test resetting environment."""
        judge = RuleBasedJudge()
        env = DebateEnvironment(judge)

        obs = env.reset()

        assert not obs.done
        assert len(obs.messages) > 0
        # Should contain debate topic and position
        content = " ".join(msg.content for msg in obs.messages)
        assert "PRO" in content or "CON" in content

    def test_environment_step_pro_argument(self):
        """Test stepping with PRO argument."""
        judge = RuleBasedJudge()
        env = DebateEnvironment(judge, max_turns=3)

        obs = env.reset()

        # Provide PRO argument
        action = Message(
            role=MessageRole.ASSISTANT,
            content="I argue that AI brings significant benefits to society.",
        )

        obs, reward = env.step(action)

        assert reward.value >= 0
        assert not obs.done or env.current_turn >= env.max_turns

    def test_environment_max_turns(self):
        """Test environment terminates after max turns."""
        judge = RuleBasedJudge()
        env = DebateEnvironment(judge, max_turns=2)

        env.reset()

        action = Message(role=MessageRole.ASSISTANT, content="Argument 1")
        env.step(action)

        action = Message(role=MessageRole.ASSISTANT, content="Argument 2")
        obs, reward = env.step(action)

        assert obs.done
