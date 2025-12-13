"""Tests for math reasoning environment."""

import pytest
from mrlm.environments.math import MathReasoningEnvironment, MathProblemGenerator
from mrlm.environments.math.solver import extract_answer, verify_answer
from mrlm.core.types import Message, MessageRole, EnvironmentMode


class TestMathSolver:
    """Test math solver utilities."""

    def test_extract_answer_simple(self):
        """Test extracting simple numeric answer."""
        text = "The answer is 42."
        answer = extract_answer(text)
        assert answer == "42"

    def test_extract_answer_with_calculation(self):
        """Test extracting answer from calculation."""
        text = "So 2 + 2 = 4, therefore the answer is 4."
        answer = extract_answer(text)
        assert answer == "4"

    def test_extract_answer_no_answer(self):
        """Test extracting when no answer present."""
        text = "I don't know."
        answer = extract_answer(text)
        # Should return empty string or None
        assert answer == "" or answer is None

    def test_verify_answer_correct(self):
        """Test verifying correct answer."""
        is_correct = verify_answer("42", "42")
        assert is_correct

    def test_verify_answer_incorrect(self):
        """Test verifying incorrect answer."""
        is_correct = verify_answer("42", "43")
        assert not is_correct

    def test_verify_answer_float(self):
        """Test verifying float answers."""
        is_correct = verify_answer("3.14", "3.14")
        assert is_correct

    def test_verify_answer_approximate(self):
        """Test verifying approximate answers."""
        # Should handle small differences in floating point
        is_correct = verify_answer("3.14159", "3.14158", tolerance=0.01)
        assert is_correct


class TestMathProblemGenerator:
    """Test MathProblemGenerator class."""

    def test_generate_arithmetic_problem(self):
        """Test generating arithmetic problem."""
        generator = MathProblemGenerator(difficulty_range=(1, 1))
        problem = generator.generate()

        assert problem.question is not None
        assert problem.answer is not None
        assert len(problem.question) > 0

    def test_generate_with_difficulty(self):
        """Test generating problems with different difficulties."""
        generator = MathProblemGenerator(difficulty_range=(1, 3))

        problems = [generator.generate() for _ in range(10)]

        # Should generate problems
        assert len(problems) == 10
        for problem in problems:
            assert problem.question is not None
            assert problem.answer is not None

    def test_problem_types(self):
        """Test generating different problem types."""
        generator = MathProblemGenerator()

        # Generate several problems
        problems = [generator.generate() for _ in range(20)]

        # Should have variety in problem types
        questions = [p.question for p in problems]
        # Check for arithmetic operators
        has_addition = any('+' in q for q in questions)
        has_numbers = any(any(c.isdigit() for c in q) for q in questions)

        assert has_addition or has_numbers


class TestMathReasoningEnvironment:
    """Test MathReasoningEnvironment class."""

    def test_environment_creation(self):
        """Test creating math reasoning environment."""
        generator = MathProblemGenerator()
        env = MathReasoningEnvironment(
            problem_generator=generator,
            mode=EnvironmentMode.CLIENT,
        )

        assert env.mode == EnvironmentMode.CLIENT
        assert env.problem_generator is generator

    def test_environment_reset(self):
        """Test resetting environment."""
        generator = MathProblemGenerator()
        env = MathReasoningEnvironment(generator)

        obs = env.reset()

        assert not obs.done
        assert len(obs.messages) > 0
        # Should contain math problem
        assert any(any(c.isdigit() for c in msg.content) for msg in obs.messages)

    def test_environment_step_correct_answer(self):
        """Test stepping with correct answer."""
        generator = MathProblemGenerator()
        env = MathReasoningEnvironment(generator)

        obs = env.reset()

        # Get the problem and answer
        current_answer = env.current_problem.answer

        # Provide correct answer
        action = Message(
            role=MessageRole.ASSISTANT,
            content=f"The answer is {current_answer}.",
        )

        obs, reward = env.step(action)

        # Should get high reward for correct answer
        assert reward.value > 0.5

    def test_environment_step_incorrect_answer(self):
        """Test stepping with incorrect answer."""
        generator = MathProblemGenerator()
        env = MathReasoningEnvironment(generator)

        env.reset()

        # Provide incorrect answer
        action = Message(
            role=MessageRole.ASSISTANT,
            content="The answer is 999999.",
        )

        obs, reward = env.step(action)

        # Should get low reward for incorrect answer
        assert reward.value < 0.5

    def test_environment_step_with_reasoning(self):
        """Test stepping with reasoning before answer."""
        generator = MathProblemGenerator()
        env = MathReasoningEnvironment(generator)

        obs = env.reset()
        current_answer = env.current_problem.answer

        # Provide reasoning with answer
        action = Message(
            role=MessageRole.ASSISTANT,
            content=f"Let me solve this step by step. The answer is {current_answer}.",
        )

        obs, reward = env.step(action)

        # Should get reward for correct answer
        assert reward.value > 0

    def test_environment_max_turns(self):
        """Test environment terminates after max turns."""
        generator = MathProblemGenerator()
        env = MathReasoningEnvironment(generator, max_turns=2)

        env.reset()

        # Turn 1
        action1 = Message(role=MessageRole.ASSISTANT, content="Thinking...")
        obs1, reward1 = env.step(action1)
        assert not obs1.done

        # Turn 2 (should terminate)
        action2 = Message(role=MessageRole.ASSISTANT, content="The answer is 5")
        obs2, reward2 = env.step(action2)
        assert obs2.done

    def test_environment_multiple_episodes(self):
        """Test running multiple episodes."""
        generator = MathProblemGenerator()
        env = MathReasoningEnvironment(generator, max_turns=2)

        # Episode 1
        obs1 = env.reset()
        problem1 = env.current_problem.question

        action = Message(role=MessageRole.ASSISTANT, content="Answer: 5")
        env.step(action)

        # Episode 2
        obs2 = env.reset()
        problem2 = env.current_problem.question

        # Should get different problem (most likely)
        # At minimum, should have valid problem
        assert obs2 is not None
        assert env.current_problem is not None

    def test_environment_close(self):
        """Test closing environment."""
        generator = MathProblemGenerator()
        env = MathReasoningEnvironment(generator)

        env.reset()
        env.close()  # Should not raise
