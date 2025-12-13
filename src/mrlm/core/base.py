"""
Base environment class for MRLM.

This module defines the BaseEnvironment abstract class that all environments
(both LLM and simulated) must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

from mrlm.core.types import Message, Observation, Reward, EnvironmentMode


class BaseEnvironment(ABC):
    """
    Abstract base class for all environments in MRLM.

    All environments (both LLM environments and simulated environments)
    must implement this interface. This unified interface allows trainers
    to work with any type of environment without knowing its internal implementation.

    Environments can operate in two modes:
    - SERVER: Environment is being trained (gradients computed, parameters updated)
    - CLIENT: Environment is frozen for inference only (no gradients)

    Attributes:
        mode: Current operation mode (SERVER or CLIENT)
        _state: Internal state dictionary

    Example:
        >>> class MyEnvironment(BaseEnvironment):
        ...     def reset(self):
        ...         return Observation(messages=[], done=False)
        ...
        ...     def step(self, action):
        ...         obs = Observation(messages=[action], done=True)
        ...         reward = Reward(value=1.0)
        ...         return obs, reward
        ...
        ...     def close(self):
        ...         pass
    """

    def __init__(self, mode: EnvironmentMode = EnvironmentMode.CLIENT):
        """
        Initialize base environment.

        Args:
            mode: Operation mode (SERVER or CLIENT). Defaults to CLIENT.
        """
        self.mode = mode
        self._state: Dict[str, Any] = {}

    @abstractmethod
    def reset(self) -> Observation:
        """
        Reset environment to initial state.

        This method should:
        1. Reset internal state
        2. Generate initial observation
        3. Return the observation

        Returns:
            Initial observation with done=False

        Example:
            >>> env = MyEnvironment()
            >>> obs = env.reset()
            >>> assert not obs.done
        """
        pass

    @abstractmethod
    def step(self, action: Message) -> Tuple[Observation, Reward]:
        """
        Take an action in the environment and return result.

        This method should:
        1. Process the action message
        2. Update internal state
        3. Compute reward
        4. Generate next observation
        5. Return (observation, reward)

        Args:
            action: Message containing the action to take

        Returns:
            Tuple of (observation, reward)
                - observation: New state observation
                - reward: Reward signal for the action

        Example:
            >>> env = MyEnvironment()
            >>> env.reset()
            >>> action = Message(role=MessageRole.ASSISTANT, content="Hello")
            >>> obs, reward = env.step(action)
        """
        pass

    @abstractmethod
    def close(self):
        """
        Clean up environment resources.

        This method should release any resources held by the environment
        (e.g., network connections, file handles, subprocesses).

        Example:
            >>> env = MyEnvironment()
            >>> env.reset()
            >>> # ... use environment ...
            >>> env.close()
        """
        pass

    @property
    def is_server_mode(self) -> bool:
        """Check if environment is in SERVER mode (being trained)."""
        return self.mode == EnvironmentMode.SERVER

    @property
    def is_client_mode(self) -> bool:
        """Check if environment is in CLIENT mode (frozen for inference)."""
        return self.mode == EnvironmentMode.CLIENT

    def set_mode(self, mode: EnvironmentMode):
        """
        Change environment operation mode.

        Args:
            mode: New operation mode

        Example:
            >>> env = MyEnvironment(mode=EnvironmentMode.CLIENT)
            >>> env.set_mode(EnvironmentMode.SERVER)
            >>> assert env.is_server_mode
        """
        self.mode = mode

    def get_state(self) -> Dict[str, Any]:
        """
        Get current internal state.

        Returns:
            Dictionary containing internal state

        Note:
            This is different from the observation returned to agents.
            State contains internal environment information not exposed
            to the agent.
        """
        return self._state.copy()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures resources are cleaned up."""
        self.close()
        return False

    def __repr__(self) -> str:
        """String representation of environment."""
        return f"{self.__class__.__name__}(mode={self.mode.value})"
