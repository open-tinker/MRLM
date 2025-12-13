"""Tests for code execution environment."""

import pytest
from mrlm.environments.code import CodeExecutionEnvironment, CodeProblemGenerator
from mrlm.environments.code.executor import CodeExecutor
from mrlm.core.types import Message, MessageRole, EnvironmentMode


class TestCodeExecutor:
    """Test CodeExecutor class."""

    def test_execute_simple_code(self):
        """Test executing simple Python code."""
        executor = CodeExecutor(timeout=5.0)

        code = "print('Hello, World!')"
        result = executor.execute(code)

        assert result.success
        assert "Hello, World!" in result.output

    def test_execute_with_test_cases(self):
        """Test code with test cases."""
        executor = CodeExecutor(timeout=5.0)

        code = """
def add(a, b):
    return a + b
"""
        test_cases = [
            ("assert add(2, 3) == 5", True),
            ("assert add(0, 0) == 0", True),
            ("assert add(-1, 1) == 0", True),
        ]

        result = executor.execute_with_tests(code, test_cases)

        assert result.success
        assert result.tests_passed == 3

    def test_execute_syntax_error(self):
        """Test executing code with syntax error."""
        executor = CodeExecutor(timeout=5.0)

        code = "print('unclosed string"
        result = executor.execute(code)

        assert not result.success
        assert "SyntaxError" in result.error or "syntax" in result.error.lower()

    def test_execute_runtime_error(self):
        """Test executing code with runtime error."""
        executor = CodeExecutor(timeout=5.0)

        code = "x = 1 / 0"
        result = executor.execute(code)

        assert not result.success
        assert "ZeroDivisionError" in result.error or "division" in result.error.lower()

    @pytest.mark.slow
    def test_execute_timeout(self):
        """Test code execution timeout."""
        executor = CodeExecutor(timeout=0.5)

        # Infinite loop
        code = "while True: pass"
        result = executor.execute(code)

        assert not result.success
        assert "timeout" in result.error.lower() or "time" in result.error.lower()


class TestCodeProblemGenerator:
    """Test CodeProblemGenerator class."""

    def test_generate_problem(self):
        """Test generating a code problem."""
        generator = CodeProblemGenerator()
        problem = generator.generate()

        assert problem.description is not None
        assert len(problem.description) > 0
        assert problem.test_cases is not None
        assert len(problem.test_cases) > 0

    def test_generate_multiple_problems(self):
        """Test generating multiple unique problems."""
        generator = CodeProblemGenerator()

        problems = [generator.generate() for _ in range(5)]

        # Should generate problems (may or may not be unique)
        assert len(problems) == 5
        for problem in problems:
            assert problem.description is not None


class TestCodeExecutionEnvironment:
    """Test CodeExecutionEnvironment class."""

    def test_environment_creation(self):
        """Test creating code execution environment."""
        generator = CodeProblemGenerator()
        env = CodeExecutionEnvironment(
            problem_generator=generator,
            mode=EnvironmentMode.CLIENT,
        )

        assert env.mode == EnvironmentMode.CLIENT
        assert env.problem_generator is generator

    def test_environment_reset(self):
        """Test resetting environment."""
        generator = CodeProblemGenerator()
        env = CodeExecutionEnvironment(generator)

        obs = env.reset()

        assert not obs.done
        assert len(obs.messages) > 0
        # Should contain problem description
        assert any("function" in msg.content.lower() or "write" in msg.content.lower()
                   for msg in obs.messages)

    def test_environment_step_correct_solution(self):
        """Test stepping with correct solution."""
        generator = CodeProblemGenerator()
        env = CodeExecutionEnvironment(generator, max_turns=3)

        obs = env.reset()

        # Simple correct solution
        action = Message(
            role=MessageRole.ASSISTANT,
            content="""
def add(a, b):
    return a + b
""",
        )

        obs, reward = env.step(action)

        # Should get some reward for valid code
        assert reward.value >= 0

    def test_environment_step_syntax_error(self):
        """Test stepping with syntax error."""
        generator = CodeProblemGenerator()
        env = CodeExecutionEnvironment(generator)

        env.reset()

        # Invalid code
        action = Message(
            role=MessageRole.ASSISTANT,
            content="def invalid syntax here",
        )

        obs, reward = env.step(action)

        # Should get low/zero reward for invalid code
        assert reward.value <= 0.5

    def test_environment_max_turns(self):
        """Test environment terminates after max turns."""
        generator = CodeProblemGenerator()
        env = CodeExecutionEnvironment(generator, max_turns=2)

        env.reset()

        # Turn 1
        action1 = Message(role=MessageRole.ASSISTANT, content="print('test')")
        obs1, reward1 = env.step(action1)
        assert not obs1.done

        # Turn 2 (should terminate)
        action2 = Message(role=MessageRole.ASSISTANT, content="print('test2')")
        obs2, reward2 = env.step(action2)
        assert obs2.done

    def test_environment_multiple_episodes(self):
        """Test running multiple episodes."""
        generator = CodeProblemGenerator()
        env = CodeExecutionEnvironment(generator, max_turns=2)

        # Episode 1
        obs1 = env.reset()
        action = Message(role=MessageRole.ASSISTANT, content="pass")
        env.step(action)

        # Episode 2
        obs2 = env.reset()

        # Should get new problem
        assert obs2 is not None
        assert not obs2.done
