"""Tests for LLM environment."""

import pytest
import torch
from mrlm.core.llm_environment import LLMEnvironment
from mrlm.core.types import Message, Observation, EnvironmentMode, MessageRole


class TestLLMEnvironment:
    """Test LLMEnvironment class."""

    def test_llm_env_creation(self, model, tokenizer):
        """Test creating LLM environment."""
        env = LLMEnvironment(
            model=model,
            tokenizer=tokenizer,
            mode=EnvironmentMode.SERVER,
        )
        assert env.mode == EnvironmentMode.SERVER
        assert env.model is model
        assert env.tokenizer is tokenizer

    def test_llm_env_reset(self, model, tokenizer):
        """Test resetting LLM environment."""
        env = LLMEnvironment(model, tokenizer, mode=EnvironmentMode.SERVER)
        obs = env.reset()
        assert isinstance(obs, Observation)
        assert not obs.done

    def test_llm_env_generate(self, model, tokenizer):
        """Test generating response."""
        env = LLMEnvironment(model, tokenizer, mode=EnvironmentMode.SERVER)
        env.reset()

        # Create a simple prompt
        messages = [
            Message(role=MessageRole.SYSTEM, content="You are helpful."),
            Message(role=MessageRole.USER, content="Say hi"),
        ]

        # Generate response
        response = env.generate(messages, max_length=20)
        assert isinstance(response, str)
        assert len(response) > 0

    def test_llm_env_step(self, model, tokenizer):
        """Test stepping with action."""
        env = LLMEnvironment(model, tokenizer, mode=EnvironmentMode.SERVER)
        obs = env.reset()

        # Step with a message
        action = Message(role=MessageRole.USER, content="Test message")
        obs, reward = env.step(action)

        assert isinstance(obs, Observation)
        assert len(obs.messages) > 0

    def test_llm_env_generation_config(self, model, tokenizer):
        """Test with custom generation config."""
        from transformers import GenerationConfig

        gen_config = GenerationConfig(
            max_length=50,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

        env = LLMEnvironment(
            model=model,
            tokenizer=tokenizer,
            mode=EnvironmentMode.SERVER,
            generation_config=gen_config,
        )

        assert env.generation_config.max_length == 50
        assert env.generation_config.temperature == 0.7

    def test_llm_env_device(self, model, tokenizer, device):
        """Test LLM environment on device."""
        env = LLMEnvironment(model, tokenizer, mode=EnvironmentMode.SERVER)
        # Model should already be on device from fixture
        assert next(env.model.parameters()).device.type == device.type

    @pytest.mark.slow
    def test_llm_env_multiple_turns(self, model, tokenizer):
        """Test multiple turn interaction."""
        env = LLMEnvironment(model, tokenizer, mode=EnvironmentMode.SERVER)
        env.reset()

        # Multiple turns
        for i in range(3):
            action = Message(role=MessageRole.USER, content=f"Turn {i}")
            obs, reward = env.step(action)
            assert isinstance(obs, Observation)

    def test_llm_env_close(self, model, tokenizer):
        """Test closing environment."""
        env = LLMEnvironment(model, tokenizer, mode=EnvironmentMode.SERVER)
        env.reset()
        env.close()  # Should not raise

    def test_llm_env_batched_generation(self, model, tokenizer):
        """Test generating for multiple prompts."""
        env = LLMEnvironment(model, tokenizer, mode=EnvironmentMode.SERVER)

        messages_list = [
            [Message(role=MessageRole.USER, content="Prompt 1")],
            [Message(role=MessageRole.USER, content="Prompt 2")],
        ]

        # This tests that the environment can handle multiple prompts
        for messages in messages_list:
            response = env.generate(messages, max_length=20)
            assert isinstance(response, str)
