"""
Code execution environment for training LLMs on programming tasks.

This environment supports training LLMs to write Python code by providing
problems with test cases and rewarding correct solutions.
"""

import re
from typing import Dict, List, Optional, Tuple

from mrlm.core.simulated_environment import SimulatedEnvironment
from mrlm.core.types import Message, MessageRole, Observation, Reward
from mrlm.environments.code.executor import PythonExecutor


class CodeExecutionEnvironment(SimulatedEnvironment):
    """
    Environment for code generation and execution.

    This environment presents coding problems to the LLM and evaluates
    the generated code against test cases.

    Rewards are based on:
    - Proportion of test cases passed
    - Code correctness
    - Execution success

    Example:
        >>> problems = [
        ...     {
        ...         "id": "test_1",
        ...         "prompt": "Write a function that adds two numbers",
        ...         "function_name": "add",
        ...         "test_cases": [((1, 2), 3), ((0, 0), 0)],
        ...     }
        ... ]
        >>> env = CodeExecutionEnvironment(problems=problems)
        >>> obs = env.reset()
        >>> # LLM generates code
        >>> action = Message(role=MessageRole.ASSISTANT, content="def add(a, b): return a + b")
        >>> obs, reward = env.step(action)
    """

    def __init__(
        self,
        problems: Optional[List[Dict]] = None,
        dataset: Optional[str] = None,
        timeout: float = 5.0,
        max_attempts: int = 3,
    ):
        """
        Initialize code execution environment.

        Args:
            problems: List of problem dictionaries
            dataset: Name of dataset to load ("humaneval", "mbpp")
            timeout: Maximum execution time per test
            max_attempts: Maximum solution attempts before episode ends
        """
        super().__init__()

        self.timeout = timeout
        self.max_attempts = max_attempts

        # Load problems
        if problems is not None:
            self.problems = problems
        elif dataset is not None:
            self.problems = self._load_dataset(dataset)
        else:
            # Default demo problems
            self.problems = self._get_demo_problems()

        # Executor
        self.executor = PythonExecutor(timeout=timeout)

        # Episode state
        self.current_problem: Optional[Dict] = None
        self.attempt_count = 0

    def _get_demo_problems(self) -> List[Dict]:
        """Get demonstration problems for testing."""
        return [
            {
                "id": "add_numbers",
                "prompt": "Write a function called 'add' that takes two numbers and returns their sum.",
                "function_name": "add",
                "test_cases": [((1, 2), 3), ((5, 7), 12), ((0, 0), 0), ((-1, 1), 0)],
            },
            {
                "id": "reverse_string",
                "prompt": "Write a function called 'reverse' that takes a string and returns it reversed.",
                "function_name": "reverse",
                "test_cases": [
                    (("hello",), "olleh"),
                    (("world",), "dlrow"),
                    (("",), ""),
                    (("a",), "a"),
                ],
            },
            {
                "id": "count_vowels",
                "prompt": "Write a function called 'count_vowels' that counts the number of vowels (a,e,i,o,u) in a string.",
                "function_name": "count_vowels",
                "test_cases": [
                    (("hello",), 2),
                    (("world",), 1),
                    (("aeiou",), 5),
                    (("xyz",), 0),
                ],
            },
        ]

    def _load_dataset(self, dataset: str) -> List[Dict]:
        """
        Load problems from a dataset.

        Args:
            dataset: Dataset name ("humaneval", "mbpp")

        Returns:
            List of problem dictionaries

        Note:
            This is a placeholder. In production, this would load
            from actual datasets like HumanEval or MBPP.
        """
        # Placeholder - would integrate with datasets library
        if dataset.lower() == "humaneval":
            # Would load HumanEval dataset
            return self._get_demo_problems()
        elif dataset.lower() == "mbpp":
            # Would load MBPP dataset
            return self._get_demo_problems()
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    def reset(self) -> Observation:
        """Reset environment with a new coding problem."""
        import random

        # Sample a problem
        self.current_problem = random.choice(self.problems)
        self.attempt_count = 0

        # Create prompt
        system_msg = Message(
            role=MessageRole.SYSTEM,
            content="You are an expert Python programmer. Write clean, correct code.",
        )

        user_msg = Message(
            role=MessageRole.USER,
            content=f"Problem: {self.current_problem['prompt']}\n\n"
            f"Write a Python function that solves this problem. "
            f"Include only the function definition, no examples or test code.",
        )

        return Observation(
            messages=[system_msg, user_msg],
            state={"problem_id": self.current_problem["id"]},
            done=False,
        )

    def step(self, action: Message) -> Tuple[Observation, Reward]:
        """
        Execute the LLM's code and return feedback.

        Args:
            action: Message containing code from LLM

        Returns:
            Tuple of (observation, reward)
        """
        self.attempt_count += 1

        # Parse code from message
        code = self._parse_action(action)

        # Execute code against test cases
        test_results = self.executor.test_function(
            code=code,
            function_name=self.current_problem["function_name"],
            test_cases=self.current_problem["test_cases"],
        )

        # Compute reward
        reward = self._compute_reward(code, test_results)

        # Create feedback
        feedback = self._create_feedback(test_results)

        # Check if episode is done
        done = test_results["all_passed"] or self.attempt_count >= self.max_attempts

        # Create observation
        feedback_msg = Message(role=MessageRole.ENVIRONMENT, content=feedback)

        obs = Observation(
            messages=[action, feedback_msg],
            state={"problem_id": self.current_problem["id"], "attempt": self.attempt_count},
            done=done,
            info=test_results,
        )

        return obs, reward

    def _parse_action(self, message: Message) -> str:
        """
        Extract Python code from message.

        Handles markdown code blocks and plain text.

        Args:
            message: Message from LLM

        Returns:
            Extracted Python code
        """
        content = message.content

        # Try to extract from markdown code block
        code_block_pattern = r"```python\s*(.*?)\s*```"
        matches = re.findall(code_block_pattern, content, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Try without language specifier
        code_block_pattern = r"```\s*(.*?)\s*```"
        matches = re.findall(code_block_pattern, content, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Use entire content as code
        return content.strip()

    def _compute_reward(self, code: str, test_results: Dict) -> Reward:
        """
        Compute reward based on test results.

        Reward components:
        - base: Proportion of tests passed (0.0 to 1.0)
        - bonus: +1.0 for passing all tests
        - penalty: -0.5 for syntax/runtime errors

        Args:
            code: Submitted code
            test_results: Results from test execution

        Returns:
            Reward with value and components
        """
        passed = test_results["passed"]
        total = test_results["total"]

        # Base reward: proportion of tests passed
        base_reward = passed / total if total > 0 else 0.0

        # Bonus for passing all tests
        bonus = 1.0 if test_results["all_passed"] else 0.0

        # Check for errors in test results
        has_errors = any(
            not result["success"] for result in test_results["test_results"]
        )
        error_penalty = -0.5 if has_errors else 0.0

        # Total reward
        total_reward = base_reward + bonus + error_penalty

        return Reward(
            value=total_reward,
            components={
                "base": base_reward,
                "bonus": bonus,
                "error_penalty": error_penalty,
                "tests_passed": passed,
                "tests_total": total,
            },
            info=test_results,
        )

    def _create_feedback(self, test_results: Dict) -> str:
        """
        Create human-readable feedback from test results.

        Args:
            test_results: Results from test execution

        Returns:
            Feedback string
        """
        passed = test_results["passed"]
        total = test_results["total"]

        feedback = f"Test Results: {passed}/{total} tests passed\n"

        if test_results["all_passed"]:
            feedback += "✓ All tests passed! Great job!\n"
        else:
            feedback += "\nFailing tests:\n"
            for result in test_results["test_results"]:
                if not result["passed"]:
                    feedback += f"  - Test {result['test_id']}: "
                    feedback += f"Input: {result['inputs']}, "
                    feedback += f"Expected: {result['expected']}, "
                    if "actual" in result:
                        feedback += f"Got: {result['actual']}\n"
                    elif "error" in result:
                        feedback += f"Error: {result['error'][:100]}...\n"

        return feedback

    def close(self):
        """Clean up resources."""
        pass

    def __repr__(self) -> str:
        """String representation."""
        return f"CodeExecutionEnvironment(problems={len(self.problems)}, timeout={self.timeout}s)"
