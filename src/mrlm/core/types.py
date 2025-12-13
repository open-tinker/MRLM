"""
Core type definitions for MRLM.

This module defines the fundamental data structures used throughout the library:
- Message: Individual messages in conversations
- Observation: State observations from environments
- Reward: Reward signals with optional decomposition
- RolloutBatch: Batched training data
- Enums for modes and roles
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import torch


class EnvironmentMode(Enum):
    """
    Environment operation mode.

    SERVER: Environment is being trained (model parameters updated)
    CLIENT: Environment is frozen for inference only
    """

    SERVER = "server"
    CLIENT = "client"


class MessageRole(Enum):
    """Role of message sender in conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    ENVIRONMENT = "environment"


@dataclass
class Message:
    """
    A single message in a conversation.

    Attributes:
        role: Who sent the message (system, user, assistant, environment)
        content: The text content of the message
        metadata: Optional metadata (e.g., log probs, token IDs, timing info)

    Example:
        >>> msg = Message(role=MessageRole.USER, content="Hello, world!")
        >>> msg = Message(
        ...     role=MessageRole.ASSISTANT,
        ...     content="Hi there!",
        ...     metadata={"log_probs": [-0.1, -0.2, -0.15]}
        ... )
    """

    role: Union[MessageRole, str]
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Convert string role to MessageRole enum if needed."""
        if isinstance(self.role, str):
            self.role = MessageRole(self.role)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "role": self.role.value if isinstance(self.role, MessageRole) else self.role,
            "content": self.content,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create Message from dictionary."""
        return cls(
            role=data["role"], content=data["content"], metadata=data.get("metadata", {})
        )


@dataclass
class Observation:
    """
    Observation returned by environment after taking an action.

    Attributes:
        messages: List of messages representing the current conversation state
        state: Optional internal environment state (not shown to agents)
        done: Whether the episode has terminated
        info: Additional information about the observation

    Example:
        >>> obs = Observation(
        ...     messages=[Message(role=MessageRole.ASSISTANT, content="Solution: 42")],
        ...     done=True,
        ...     info={"steps": 5}
        ... )
    """

    messages: List[Message]
    state: Optional[Dict[str, Any]] = None
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "messages": [msg.to_dict() for msg in self.messages],
            "state": self.state,
            "done": self.done,
            "info": self.info,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observation":
        """Create Observation from dictionary."""
        return cls(
            messages=[Message.from_dict(msg) for msg in data["messages"]],
            state=data.get("state"),
            done=data.get("done", False),
            info=data.get("info", {}),
        )


@dataclass
class Reward:
    """
    Reward signal from environment.

    Attributes:
        value: Total reward value (float)
        components: Optional decomposition of reward into components
                   (e.g., {"correctness": 0.8, "efficiency": 0.2})
        info: Additional information about the reward computation

    Example:
        >>> reward = Reward(value=1.0)
        >>> reward = Reward(
        ...     value=0.75,
        ...     components={"correctness": 0.5, "style": 0.25},
        ...     info={"tests_passed": 3, "tests_total": 4}
        ... )
    """

    value: float
    components: Dict[str, float] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "value": self.value,
            "components": self.components,
            "info": self.info,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Reward":
        """Create Reward from dictionary."""
        return cls(
            value=data["value"],
            components=data.get("components", {}),
            info=data.get("info", {}),
        )


@dataclass
class RolloutBatch:
    """
    Batch of rollout data for training.

    This dataclass stores trajectories collected during environment interaction
    for use in RL training algorithms.

    Attributes:
        observations: List of observations from environment
        actions: List of actions taken (as Messages)
        rewards: List of rewards received
        values: Optional value estimates from critic (for PPO)
        log_probs: Optional log probabilities of actions (for PPO)
        advantages: Optional computed advantages (for PPO)
        returns: Optional computed returns/targets (for PPO)
        dones: Optional episode termination flags
        attention_masks: Optional attention masks for transformer models

    Example:
        >>> batch = RolloutBatch(
        ...     observations=[obs1, obs2, obs3],
        ...     actions=[action1, action2, action3],
        ...     rewards=[reward1, reward2, reward3],
        ...     log_probs=torch.tensor([-0.5, -0.6, -0.4]),
        ...     values=torch.tensor([1.2, 1.5, 0.8])
        ... )
    """

    observations: List[Observation]
    actions: List[Message]
    rewards: List[Reward]
    values: Optional[torch.Tensor] = None
    log_probs: Optional[torch.Tensor] = None
    advantages: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None
    dones: Optional[torch.Tensor] = None
    attention_masks: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        """Return number of samples in batch."""
        return len(self.observations)

    def to_device(self, device: torch.device) -> "RolloutBatch":
        """Move all tensors to specified device."""
        return RolloutBatch(
            observations=self.observations,
            actions=self.actions,
            rewards=self.rewards,
            values=self.values.to(device) if self.values is not None else None,
            log_probs=self.log_probs.to(device) if self.log_probs is not None else None,
            advantages=self.advantages.to(device) if self.advantages is not None else None,
            returns=self.returns.to(device) if self.returns is not None else None,
            dones=self.dones.to(device) if self.dones is not None else None,
            attention_masks=(
                self.attention_masks.to(device) if self.attention_masks is not None else None
            ),
        )


@dataclass
class GenerationConfig:
    """
    Configuration for text generation.

    Attributes:
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling threshold
        do_sample: Whether to use sampling (vs greedy decoding)
        num_beams: Number of beams for beam search
        repetition_penalty: Penalty for repeating tokens
        length_penalty: Penalty for sequence length
        eos_token_id: End-of-sequence token ID
        pad_token_id: Padding token ID
    """

    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    num_beams: int = 1
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for use with transformers."""
        config = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "do_sample": self.do_sample,
            "num_beams": self.num_beams,
            "repetition_penalty": self.repetition_penalty,
            "length_penalty": self.length_penalty,
        }
        if self.eos_token_id is not None:
            config["eos_token_id"] = self.eos_token_id
        if self.pad_token_id is not None:
            config["pad_token_id"] = self.pad_token_id
        return config
