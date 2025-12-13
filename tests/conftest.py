"""Pytest configuration and shared fixtures."""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_model_name():
    """Name of a small model for testing."""
    # Use a very small model for fast tests
    return "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"


@pytest.fixture
def model(small_model_name, device):
    """Load a small model for testing."""
    model = AutoModelForCausalLM.from_pretrained(
        small_model_name,
        torch_dtype=torch.float32,
    )
    return model.to(device)


@pytest.fixture
def tokenizer(small_model_name):
    """Load tokenizer for testing."""
    tokenizer = AutoTokenizer.from_pretrained(small_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    from mrlm.core.types import Message, MessageRole

    return [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="What is 2+2?"),
        Message(role=MessageRole.ASSISTANT, content="2+2 equals 4."),
    ]


@pytest.fixture
def sample_observation(sample_messages):
    """Sample observation for testing."""
    from mrlm.core.types import Observation

    return Observation(
        messages=sample_messages,
        state={"step": 1},
        done=False,
        info={"test": True},
    )


@pytest.fixture
def sample_reward():
    """Sample reward for testing."""
    from mrlm.core.types import Reward

    return Reward(
        value=1.0,
        done=False,
        info={"success": True},
    )


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test."""
    torch.manual_seed(42)
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
