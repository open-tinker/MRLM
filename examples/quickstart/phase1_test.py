"""
Phase 1 Test: Basic LLM Environment

This script tests the core functionality implemented in Phase 1:
- Loading a model with HFModelWrapper
- Creating an LLMEnvironment
- Running basic inference

Note: This requires transformers and torch to be installed.
For testing, we use a small model (gpt2).
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mrlm.core import LLMEnvironment, EnvironmentMode, Message, MessageRole
from mrlm.models import HFModelWrapper


def test_llm_environment():
    """Test basic LLM environment functionality."""
    print("=" * 60)
    print("Phase 1 Test: LLM Environment")
    print("=" * 60)

    # Load model (using small model for testing)
    print("\n1. Loading model...")
    model_name = "gpt2"  # Small model for testing
    wrapper = HFModelWrapper(model_name, device="cpu")  # Use CPU for compatibility
    print(f"   ✓ Loaded {model_name}")

    # Create LLM environment in client mode
    print("\n2. Creating LLM environment (CLIENT mode)...")
    env = LLMEnvironment(
        model=wrapper.model,
        tokenizer=wrapper.tokenizer,
        mode=EnvironmentMode.CLIENT,
        system_prompt="You are a helpful assistant.",
    )
    print(f"   ✓ Created {env}")

    # Reset environment
    print("\n3. Resetting environment...")
    obs = env.reset()
    print(f"   ✓ Reset complete. Conversation has {len(obs.messages)} messages")
    if obs.messages:
        print(f"   System prompt: {obs.messages[0].content[:50]}...")

    # Send a message and get response
    print("\n4. Sending message to LLM...")
    user_message = Message(role=MessageRole.USER, content="Hello! What is 2+2?")
    print(f"   User: {user_message.content}")

    obs, reward = env.step(user_message)
    print(f"   ✓ Received response")
    print(f"   Assistant: {obs.messages[-1].content[:100]}...")
    print(f"   Conversation length: {len(obs.messages)} messages")
    print(f"   Reward: {reward.value}")

    # Test conversation history
    print("\n5. Checking conversation history...")
    conversation = env.get_conversation()
    print(f"   ✓ Conversation has {len(conversation)} messages")
    for i, msg in enumerate(conversation):
        role = msg.role.value if hasattr(msg.role, 'value') else msg.role
        content_preview = msg.content[:40] + "..." if len(msg.content) > 40 else msg.content
        print(f"   [{i}] {role}: {content_preview}")

    # Test mode switching
    print("\n6. Testing mode switching...")
    print(f"   Current mode: {env.mode.value}")
    env.set_mode(EnvironmentMode.SERVER)
    print(f"   ✓ Switched to: {env.mode.value}")
    env.set_mode(EnvironmentMode.CLIENT)
    print(f"   ✓ Switched back to: {env.mode.value}")

    # Clean up
    print("\n7. Cleaning up...")
    env.close()
    print("   ✓ Environment closed")

    print("\n" + "=" * 60)
    print("✓ Phase 1 Test PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_llm_environment()
    except Exception as e:
        print(f"\n✗ Test FAILED with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
