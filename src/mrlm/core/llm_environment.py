"""
LLM Environment - wraps language models as environments.

This module provides the LLMEnvironment class that wraps transformer models
and allows them to be used as environments in the MRLM framework.
"""

from typing import Callable, List, Optional, Tuple, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from mrlm.core.base import BaseEnvironment
from mrlm.core.types import (
    EnvironmentMode,
    GenerationConfig,
    Message,
    MessageRole,
    Observation,
    Reward,
)
from mrlm.models.generation import generate_response, generate_with_log_probs


class LLMEnvironment(BaseEnvironment):
    """
    Environment wrapper for Language Models.

    This class wraps a transformer model and treats it as an environment.
    It can operate in two modes:
    - SERVER mode: Model is being trained (gradients computed)
    - CLIENT mode: Model is frozen for inference only

    The LLM environment:
    1. Receives messages as actions
    2. Generates responses using the model
    3. Optionally computes rewards using a reward function
    4. Returns observations and rewards

    Attributes:
        model: The transformer model
        tokenizer: The tokenizer
        mode: SERVER (training) or CLIENT (inference)
        generation_config: Configuration for text generation
        reward_fn: Optional function to compute rewards
        conversation_history: Current conversation

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> env = LLMEnvironment(model, tokenizer, mode=EnvironmentMode.CLIENT)
        >>> obs = env.reset()
        >>> action = Message(role=MessageRole.USER, content="Hello!")
        >>> obs, reward = env.step(action)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        mode: EnvironmentMode = EnvironmentMode.CLIENT,
        generation_config: Optional[Union[GenerationConfig, dict]] = None,
        reward_fn: Optional[Callable[[List[Message]], Reward]] = None,
        system_prompt: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize LLM environment.

        Args:
            model: Pre-trained transformer model
            tokenizer: Tokenizer for the model
            mode: Operation mode (SERVER for training, CLIENT for inference)
            generation_config: Text generation configuration
            reward_fn: Optional function to compute rewards from conversation
            system_prompt: Optional system prompt to prepend to conversations
            device: Device to run model on (if None, uses model's current device)
        """
        super().__init__(mode=mode)

        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device

        # Generation configuration
        if generation_config is None:
            self.generation_config = GenerationConfig()
        elif isinstance(generation_config, dict):
            self.generation_config = GenerationConfig(**generation_config)
        else:
            self.generation_config = generation_config

        # Set EOS and PAD token IDs if not set
        if self.generation_config.eos_token_id is None:
            self.generation_config.eos_token_id = tokenizer.eos_token_id
        if self.generation_config.pad_token_id is None:
            self.generation_config.pad_token_id = tokenizer.pad_token_id

        # Reward function
        self.reward_fn = reward_fn

        # System prompt
        self.system_prompt = system_prompt

        # Conversation state
        self.conversation_history: List[Message] = []

        # Freeze model in client mode
        if self.is_client_mode:
            self._freeze_model()

    def _freeze_model(self):
        """Freeze model parameters for inference-only mode."""
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def _unfreeze_model(self):
        """Unfreeze model parameters for training mode."""
        for param in self.model.parameters():
            param.requires_grad = True

    def reset(self) -> Observation:
        """
        Reset conversation to initial state.

        Clears conversation history and optionally adds system prompt.

        Returns:
            Initial observation with empty or system-prompted conversation

        Example:
            >>> env = LLMEnvironment(model, tokenizer)
            >>> obs = env.reset()
            >>> assert len(obs.messages) == 0 or obs.messages[0].role == MessageRole.SYSTEM
        """
        self.conversation_history = []

        # Add system prompt if provided
        if self.system_prompt:
            system_msg = Message(
                role=MessageRole.SYSTEM, content=self.system_prompt, metadata={}
            )
            self.conversation_history.append(system_msg)

        return Observation(messages=self.conversation_history.copy(), done=False, info={})

    def step(self, action: Message) -> Tuple[Observation, Reward]:
        """
        Process input message and generate response.

        In SERVER mode:
        - Computes gradients
        - Returns log probabilities and value estimates
        - Model parameters can be updated

        In CLIENT mode:
        - Inference only, no gradients
        - Faster generation
        - Model parameters frozen

        Args:
            action: Input message from user or another environment

        Returns:
            Tuple of (observation, reward)
                - observation: Contains conversation including generated response
                - reward: Reward signal (if reward_fn provided, else 0.0)

        Example:
            >>> env = LLMEnvironment(model, tokenizer, mode=EnvironmentMode.CLIENT)
            >>> env.reset()
            >>> action = Message(role=MessageRole.USER, content="What is 2+2?")
            >>> obs, reward = env.step(action)
            >>> print(obs.messages[-1].content)  # LLM's response
        """
        # Add action to conversation history
        self.conversation_history.append(action)

        # Generate response based on mode
        if self.is_server_mode:
            response_text, log_probs, value = self._generate_with_grads()
        else:
            response_text = self._generate_inference()
            log_probs, value = None, None

        # Create response message
        response_msg = Message(
            role=MessageRole.ASSISTANT,
            content=response_text,
            metadata={"log_probs": log_probs, "value": value},
        )
        self.conversation_history.append(response_msg)

        # Compute reward if function provided
        if self.reward_fn is not None:
            reward = self.reward_fn(self.conversation_history)
        else:
            reward = Reward(value=0.0)

        # Create observation
        obs = Observation(
            messages=self.conversation_history.copy(),
            done=False,
            info={"log_probs": log_probs, "value": value, "response_length": len(response_text)},
        )

        return obs, reward

    def _generate_with_grads(self) -> Tuple[str, torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate response in SERVER mode (with gradients).

        Returns:
            Tuple of (response_text, log_probs, value)
        """
        return generate_with_log_probs(
            model=self.model,
            tokenizer=self.tokenizer,
            messages=self.conversation_history,
            generation_config=self.generation_config,
            device=self.device,
            return_value=True,
        )

    def _generate_inference(self) -> str:
        """
        Generate response in CLIENT mode (inference only).

        Returns:
            Generated response text
        """
        return generate_response(
            model=self.model,
            tokenizer=self.tokenizer,
            messages=self.conversation_history,
            generation_config=self.generation_config,
            device=self.device,
        )

    def set_mode(self, mode: EnvironmentMode):
        """
        Change environment operation mode.

        When switching to CLIENT mode, freezes model.
        When switching to SERVER mode, unfreezes model.

        Args:
            mode: New operation mode

        Example:
            >>> env = LLMEnvironment(model, tokenizer, mode=EnvironmentMode.CLIENT)
            >>> env.set_mode(EnvironmentMode.SERVER)  # Switch to training mode
        """
        super().set_mode(mode)

        if self.is_client_mode:
            self._freeze_model()
        else:
            self._unfreeze_model()

    def get_conversation(self) -> List[Message]:
        """
        Get current conversation history.

        Returns:
            List of messages in conversation
        """
        return self.conversation_history.copy()

    def set_conversation(self, messages: List[Message]):
        """
        Set conversation history directly.

        Useful for resuming conversations or providing context.

        Args:
            messages: List of messages to set as conversation

        Example:
            >>> env = LLMEnvironment(model, tokenizer)
            >>> prior_conversation = [...]
            >>> env.set_conversation(prior_conversation)
        """
        self.conversation_history = messages.copy()

    def set_reward_function(self, reward_fn: Callable[[List[Message]], Reward]):
        """
        Set or update reward function.

        Args:
            reward_fn: Function that takes conversation and returns Reward

        Example:
            >>> def my_reward(conversation):
            ...     # Check if last message contains "correct"
            ...     if "correct" in conversation[-1].content.lower():
            ...         return Reward(value=1.0)
            ...     return Reward(value=0.0)
            >>> env.set_reward_function(my_reward)
        """
        self.reward_fn = reward_fn

    def close(self):
        """
        Clean up resources.

        For LLM environments, this doesn't do much, but included for consistency.
        """
        pass

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LLMEnvironment(model={self.model.__class__.__name__}, "
            f"mode={self.mode.value}, "
            f"conversation_length={len(self.conversation_history)})"
        )
