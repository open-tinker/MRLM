"""
Math reasoning environment for training LLMs on mathematical problem solving.

This environment supports training LLMs on math word problems like GSM8K and MATH dataset.
"""

import re
from typing import Dict, List, Optional, Tuple

from mrlm.core.simulated_environment import SimulatedEnvironment
from mrlm.core.types import Message, MessageRole, Observation, Reward


class MathReasoningEnvironment(SimulatedEnvironment):
    """
    Environment for mathematical reasoning tasks.

    This environment presents math word problems and evaluates the LLM's
    numerical answers.

    Supports:
    - GSM8K-style grade school math problems
    - MATH dataset problems
    - Custom math problems

    Example:
        >>> problems = [
        ...     {
        ...         "question": "If John has 5 apples and buys 3 more, how many does he have?",
        ...         "answer": 8,
        ...         "solution": "5 + 3 = 8"
        ...     }
        ... ]
        >>> env = MathReasoningEnvironment(problems=problems)
        >>> obs = env.reset()
        >>> action = Message(role=MessageRole.ASSISTANT, content="The answer is 8.")
        >>> obs, reward = env.step(action)
    """

    def __init__(
        self,
        problems: Optional[List[Dict]] = None,
        dataset: Optional[str] = None,
        max_attempts: int = 2,
        require_solution: bool = False,
    ):
        """
        Initialize math reasoning environment.

        Args:
            problems: List of problem dictionaries with 'question' and 'answer'
            dataset: Dataset to load ("gsm8k", "math")
            max_attempts: Maximum solution attempts per problem
            require_solution: Whether to require step-by-step solution
        """
        super().__init__()

        self.max_attempts = max_attempts
        self.require_solution = require_solution

        # Load problems
        if problems is not None:
            self.problems = problems
        elif dataset is not None:
            self.problems = self._load_dataset(dataset)
        else:
            self.problems = self._get_demo_problems()

        # Episode state
        self.current_problem: Optional[Dict] = None
        self.attempt_count = 0

    def _get_demo_problems(self) -> List[Dict]:
        """Get demonstration math problems."""
        return [
            {
                "id": "addition_1",
                "question": "Sarah has 15 apples. She buys 7 more apples at the store. How many apples does Sarah have now?",
                "answer": 22,
                "solution": "15 + 7 = 22",
                "difficulty": "easy",
            },
            {
                "id": "multiplication_1",
                "question": "A box contains 8 rows of chocolates with 6 chocolates in each row. How many chocolates are in the box?",
                "answer": 48,
                "solution": "8 × 6 = 48",
                "difficulty": "easy",
            },
            {
                "id": "multi_step_1",
                "question": "Tom has $50. He buys 3 books for $12 each. How much money does Tom have left?",
                "answer": 14,
                "solution": "3 × $12 = $36 spent. $50 - $36 = $14 remaining",
                "difficulty": "medium",
            },
            {
                "id": "fractions_1",
                "question": "A pizza is cut into 8 slices. If John eats 3 slices and Mary eats 2 slices, what fraction of the pizza is left?",
                "answer": 0.375,  # 3/8 = 0.375
                "solution": "Total eaten: 3 + 2 = 5 slices. Remaining: 8 - 5 = 3 slices. Fraction: 3/8 = 0.375",
                "difficulty": "medium",
            },
        ]

    def _load_dataset(self, dataset: str) -> List[Dict]:
        """
        Load problems from dataset.

        Args:
            dataset: Dataset name ("gsm8k", "math")

        Returns:
            List of problem dictionaries

        Note:
            Placeholder - would integrate with datasets library in production
        """
        if dataset.lower() == "gsm8k":
            # Would load GSM8K dataset
            return self._get_demo_problems()
        elif dataset.lower() == "math":
            # Would load MATH dataset
            return self._get_demo_problems()
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    def reset(self) -> Observation:
        """Reset with a new math problem."""
        import random

        # Sample problem
        self.current_problem = random.choice(self.problems)
        self.attempt_count = 0

        # Create prompt
        system_msg = Message(
            role=MessageRole.SYSTEM,
            content="You are a helpful math tutor. Solve math problems step-by-step and provide the final numerical answer.",
        )

        instruction = "Solve this problem. "
        if self.require_solution:
            instruction += "Show your work step-by-step, then give the final answer."
        else:
            instruction += "Give the final numerical answer."

        user_msg = Message(
            role=MessageRole.USER,
            content=f"{self.current_problem['question']}\n\n{instruction}",
        )

        return Observation(
            messages=[system_msg, user_msg],
            state={"problem_id": self.current_problem.get("id", "unknown")},
            done=False,
        )

    def step(self, action: Message) -> Tuple[Observation, Reward]:
        """
        Evaluate the LLM's answer.

        Args:
            action: Message containing solution from LLM

        Returns:
            Tuple of (observation, reward)
        """
        self.attempt_count += 1

        # Parse answer from message
        parsed_answer = self._parse_action(action)

        # Check correctness
        is_correct = self._check_answer(parsed_answer)

        # Compute reward
        reward = self._compute_reward(action.content, parsed_answer, is_correct)

        # Create feedback
        feedback = self._create_feedback(parsed_answer, is_correct)

        # Check if done
        done = is_correct or self.attempt_count >= self.max_attempts

        # Create observation
        feedback_msg = Message(role=MessageRole.ENVIRONMENT, content=feedback)

        obs = Observation(
            messages=[action, feedback_msg],
            state={
                "problem_id": self.current_problem.get("id", "unknown"),
                "attempt": self.attempt_count,
            },
            done=done,
            info={
                "correct": is_correct,
                "parsed_answer": parsed_answer,
                "expected_answer": self.current_problem["answer"],
            },
        )

        return obs, reward

    def _parse_action(self, message: Message) -> Optional[float]:
        """
        Extract numerical answer from message.

        Looks for patterns like:
        - "The answer is 42"
        - "= 42"
        - "42"

        Args:
            message: Message from LLM

        Returns:
            Extracted numerical answer or None
        """
        content = message.content

        # Common answer patterns
        patterns = [
            r"(?:the\s+)?answer\s+is\s+([+-]?\d+(?:\.\d+)?)",  # "answer is 42"
            r"=\s*([+-]?\d+(?:\.\d+)?)\s*$",  # "= 42" at end
            r"^([+-]?\d+(?:\.\d+)?)\s*$",  # Just "42"
            r"therefore[,\s]+([+-]?\d+(?:\.\d+)?)",  # "therefore, 42"
            r"final\s+answer[:\s]+([+-]?\d+(?:\.\d+)?)",  # "final answer: 42"
        ]

        for pattern in patterns:
            match = re.search(pattern, content.lower())
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        # Try to find any number in the last line
        lines = content.strip().split("\n")
        if lines:
            last_line = lines[-1]
            numbers = re.findall(r"([+-]?\d+(?:\.\d+)?)", last_line)
            if numbers:
                try:
                    return float(numbers[-1])
                except ValueError:
                    pass

        return None

    def _check_answer(self, parsed_answer: Optional[float]) -> bool:
        """
        Check if answer is correct.

        Args:
            parsed_answer: Extracted numerical answer

        Returns:
            Whether answer is correct
        """
        if parsed_answer is None:
            return False

        expected = self.current_problem["answer"]

        # Allow small floating point tolerance
        tolerance = 1e-4
        if isinstance(expected, (int, float)):
            return abs(parsed_answer - expected) < tolerance

        return False

    def _compute_reward(
        self, full_response: str, parsed_answer: Optional[float], is_correct: bool
    ) -> Reward:
        """
        Compute reward based on answer correctness.

        Reward components:
        - correctness: +1.0 for correct, 0.0 for incorrect
        - parsing: -0.3 if answer couldn't be parsed
        - solution: +0.2 if step-by-step solution provided (bonus)

        Args:
            full_response: Full response text
            parsed_answer: Extracted answer
            is_correct: Whether answer is correct

        Returns:
            Reward with value and components
        """
        components = {}

        # Correctness reward
        correctness = 1.0 if is_correct else 0.0
        components["correctness"] = correctness

        # Parsing penalty
        if parsed_answer is None:
            parsing_penalty = -0.3
            components["parsing_penalty"] = parsing_penalty
        else:
            parsing_penalty = 0.0

        # Solution quality bonus
        if self.require_solution:
            # Check if response includes explanation/steps
            has_steps = any(
                keyword in full_response.lower()
                for keyword in ["first", "then", "therefore", "so", "because", "="]
            )
            solution_bonus = 0.2 if has_steps else 0.0
            components["solution_bonus"] = solution_bonus
        else:
            solution_bonus = 0.0

        # Total reward
        total = correctness + parsing_penalty + solution_bonus

        return Reward(value=total, components=components)

    def _create_feedback(self, parsed_answer: Optional[float], is_correct: bool) -> str:
        """
        Create feedback message.

        Args:
            parsed_answer: Extracted answer
            is_correct: Whether answer is correct

        Returns:
            Feedback string
        """
        if is_correct:
            return f"✓ Correct! The answer is {self.current_problem['answer']}."

        feedback = "✗ Incorrect. "

        if parsed_answer is None:
            feedback += "Could not parse a numerical answer from your response. "
            feedback += "Please provide a clear numerical answer."
        else:
            feedback += f"Your answer: {parsed_answer}, "
            feedback += f"Correct answer: {self.current_problem['answer']}"

        if self.attempt_count < self.max_attempts:
            feedback += "\n\nTry again!"

        return feedback

    def close(self):
        """Clean up resources."""
        pass

    def __repr__(self) -> str:
        """String representation."""
        return f"MathReasoningEnvironment(problems={len(self.problems)})"
