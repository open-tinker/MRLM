"""
Base class for simulated (non-LLM) environments.

Simulated environments are always in CLIENT mode since they are not trainable.
Examples include code execution, math verification, tool execution, etc.
"""

from abc import abstractmethod
from typing import Any, Tuple

from mrlm.core.base import BaseEnvironment
from mrlm.core.types import Message, Observation, Reward, EnvironmentMode


class SimulatedEnvironment(BaseEnvironment):
    """
    Base class for simulated (non-LLM) environments.

    Simulated environments represent tasks or verification systems that interact
    with LLMs but are not themselves trainable. They always operate in CLIENT mode.

    Examples:
    - Code execution environment (runs code, checks tests)
    - Math verification environment (checks mathematical solutions)
    - Tool execution environment (executes tool calls)
    - Game environment (plays games with LLM)

    Subclasses must implement:
    - reset(): Initialize the task/problem
    - step(): Process LLM action and return feedback
    - _parse_action(): Extract structured data from message
    - _compute_reward(): Calculate reward for action

    Example:
        >>> class CodeEnv(SimulatedEnvironment):
        ...     def reset(self):
        ...         self.problem = sample_coding_problem()
        ...         return Observation(messages=[Message(...)])
        ...
        ...     def step(self, action):
        ...         code = self._parse_action(action)
        ...         result = execute_code(code)
        ...         reward = self._compute_reward(result)
        ...         return Observation(...), reward
    """

    def __init__(self):
        """
        Initialize simulated environment.

        Simulated environments are always in CLIENT mode.
        """
        super().__init__(mode=EnvironmentMode.CLIENT)

    @abstractmethod
    def reset(self) -> Observation:
        """
        Reset environment and present new task/problem.

        Returns:
            Initial observation containing task description

        Example:
            >>> env = MySimulatedEnv()
            >>> obs = env.reset()
            >>> print(obs.messages[0].content)  # Task description
        """
        pass

    @abstractmethod
    def step(self, action: Message) -> Tuple[Observation, Reward]:
        """
        Execute action in simulated environment.

        Typical flow:
        1. Parse action message into structured format
        2. Execute action (e.g., run code, verify answer)
        3. Compute reward based on result
        4. Generate feedback observation
        5. Return (observation, reward)

        Args:
            action: Message from LLM containing action to execute

        Returns:
            Tuple of (observation, reward)

        Example:
            >>> env = MySimulatedEnv()
            >>> env.reset()
            >>> action = Message(role=MessageRole.ASSISTANT, content="solution")
            >>> obs, reward = env.step(action)
        """
        pass

    @abstractmethod
    def _parse_action(self, message: Message) -> Any:
        """
        Parse message content into actionable format.

        This method should extract structured information from the message.
        For example:
        - Extract code from markdown code blocks
        - Parse mathematical expressions
        - Extract function calls and arguments
        - Parse JSON responses

        Args:
            message: Message from LLM

        Returns:
            Parsed action in format specific to the environment

        Example:
            >>> def _parse_action(self, message):
            ...     # Extract Python code from markdown
            ...     if "```python" in message.content:
            ...         code = extract_code_block(message.content)
            ...         return code
            ...     return message.content
        """
        pass

    @abstractmethod
    def _compute_reward(self, action: Any, result: Any) -> Reward:
        """
        Compute reward based on action and execution result.

        This method should evaluate how good the action was and return
        a reward signal. Rewards can be:
        - Binary (0/1 for failure/success)
        - Continuous (0.0 to 1.0)
        - Decomposed (multiple reward components)

        Args:
            action: Parsed action
            result: Result of executing the action

        Returns:
            Reward object with value and optional components

        Example:
            >>> def _compute_reward(self, action, result):
            ...     tests_passed = result['tests_passed']
            ...     tests_total = result['tests_total']
            ...     score = tests_passed / tests_total
            ...     return Reward(
            ...         value=score,
            ...         components={'tests': score},
            ...         info=result
            ...     )
        """
        pass

    def close(self):
        """
        Clean up environment resources.

        Default implementation does nothing. Override if your environment
        needs cleanup (e.g., closing files, killing subprocesses).
        """
        pass

    def set_mode(self, mode: EnvironmentMode):
        """
        Prevent changing mode for simulated environments.

        Simulated environments must always be in CLIENT mode.
        """
        if mode != EnvironmentMode.CLIENT:
            raise ValueError(
                f"SimulatedEnvironment must be in CLIENT mode, cannot set to {mode.value}"
            )
        super().set_mode(mode)
