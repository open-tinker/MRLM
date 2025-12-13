"""
Example: Training an agent to use tools with PPO.

This example demonstrates training an LLM to:
1. Recognize when tools are needed
2. Select appropriate tools
3. Format tool calls correctly
4. Use tool results to answer questions
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mrlm.core.types import EnvironmentMode
from mrlm.core.llm_environment import LLMEnvironment
from mrlm.environments.tools import (
    ToolUseEnvironment,
    ToolRegistry,
    ToolUseTask,
    CalculatorTool,
    WebSearchTool,
    PythonREPLTool,
)
from mrlm.algorithms.ppo import PPOTrainer
from mrlm.config.training_config import (
    ExperimentConfig,
    TrainingConfig,
    ModelConfig,
    PPOConfig,
)


def create_tool_use_tasks() -> list[ToolUseTask]:
    """Create sample tasks that require tool use."""
    return [
        # Calculator tasks
        ToolUseTask(
            question="What is 15% of 280?",
            required_tools=["calculator"],
            ground_truth="42",
        ),
        ToolUseTask(
            question="If a circle has radius 5, what is its area? (Use π = 3.14159)",
            required_tools=["calculator"],
            ground_truth="78.54",
        ),
        ToolUseTask(
            question="Calculate: sqrt(144) + 3^2",
            required_tools=["calculator"],
            ground_truth="21",
        ),
        # Web search tasks
        ToolUseTask(
            question="How tall is the Eiffel Tower?",
            required_tools=["web_search"],
            ground_truth="330 meters",
        ),
        ToolUseTask(
            question="When was the first moon landing?",
            required_tools=["web_search"],
            ground_truth="july 20, 1969",
        ),
        ToolUseTask(
            question="Who created the Python programming language?",
            required_tools=["web_search"],
            ground_truth="guido van rossum",
        ),
        # Python REPL tasks
        ToolUseTask(
            question="Generate a list of the first 10 fibonacci numbers.",
            required_tools=["python_repl"],
            ground_truth="[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]",
        ),
        ToolUseTask(
            question="What is the sum of all even numbers from 1 to 20?",
            required_tools=["python_repl"],
            ground_truth="110",
        ),
        # Multi-tool tasks
        ToolUseTask(
            question="The Eiffel Tower is a certain height. If I stack 3 of them, what's the total height in meters?",
            required_tools=["web_search", "calculator"],
            ground_truth="990",
        ),
        ToolUseTask(
            question="What's the speed of light in km/h? (Speed of light is approximately 299,792,458 m/s)",
            required_tools=["calculator"],
            ground_truth="1,079,252,848.8",
        ),
    ]


def main():
    """Train LLM to use tools with PPO."""

    # Configuration
    config = ExperimentConfig(
        experiment_name="tool_use_ppo",
        training=TrainingConfig(
            algorithm="ppo",
            num_epochs=100,
            batch_size=8,
            learning_rate=5e-6,
            max_episode_length=5,  # Max 5 turns per task
            episodes_per_iteration=16,
        ),
        model=ModelConfig(
            model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
            device_map="auto",
            torch_dtype="float16",
        ),
        ppo=PPOConfig(
            clip_range=0.2,
            gamma=0.99,
            gae_lambda=0.95,
            num_ppo_epochs=4,
            value_loss_coef=0.5,
            entropy_coef=0.01,
        ),
    )

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    print(f"Loading model: {config.model.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name_or_path,
        device_map=config.model.device_map,
        torch_dtype=getattr(torch, config.model.torch_dtype),
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create policy environment (SERVER mode for training)
    policy_env = LLMEnvironment(
        model=model,
        tokenizer=tokenizer,
        mode=EnvironmentMode.SERVER,
        generation_config=config.model.generation.to_generation_config(),
    )

    # Create tool registry
    print("Setting up tools...")
    tool_registry = ToolRegistry()
    tool_registry.register(CalculatorTool())
    tool_registry.register(WebSearchTool())
    tool_registry.register(PythonREPLTool())

    print(f"Registered {len(tool_registry.tools)} tools:")
    for tool in tool_registry.list_tools():
        print(f"  - {tool.name}: {tool.description}")

    # Create tasks
    tasks = create_tool_use_tasks()
    print(f"\nCreated {len(tasks)} training tasks")

    # Create evaluation environments
    print("Creating tool use environments...")
    eval_envs = [
        ToolUseEnvironment(
            tool_registry=tool_registry,
            tasks=tasks,
            max_turns=config.training.max_episode_length,
            mode=EnvironmentMode.CLIENT,
        )
        for _ in range(4)  # 4 parallel environments
    ]

    # Create PPO trainer
    print("Initializing PPO trainer...")
    trainer = PPOTrainer(
        policy_env=policy_env,
        eval_envs=eval_envs,
        config=config,
        device=device,
    )

    # Train
    print(f"\nStarting PPO training for {config.training.num_epochs} epochs...")
    print("The agent will learn to:")
    print("  1. Recognize when tools are needed")
    print("  2. Format tool calls correctly")
    print("  3. Use tool results to answer questions")
    print()

    trainer.train(
        num_iterations=config.training.num_epochs,
        eval_every=10,
        save_every=20,
    )

    print("\nTraining complete!")

    # Demo: Test the trained model
    print("\n" + "=" * 60)
    print("Testing trained model on a sample task:")
    print("=" * 60)

    test_env = eval_envs[0]
    obs = test_env.reset()

    print(f"\nTask: {test_env.current_task.question}")
    print(f"Required tools: {test_env.current_task.required_tools}")
    print(f"Expected answer: {test_env.current_task.ground_truth}")
    print("\nAgent's attempt:")

    model.eval()
    with torch.no_grad():
        for turn in range(config.training.max_episode_length):
            # Generate response
            from mrlm.models.generation import generate_response

            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                messages=obs.messages,
                generation_config=config.model.generation.to_generation_config(),
                device=device,
            )

            print(f"\nTurn {turn + 1}:")
            print(f"Agent: {response}")

            # Take step
            from mrlm.core.types import Message, MessageRole

            action = Message(role=MessageRole.ASSISTANT, content=response)
            obs, reward = test_env.step(action)

            print(f"Reward: {reward.value:.2f}")

            if obs.done:
                print(f"\nTask completed! Final reward: {reward.value:.2f}")
                if reward.info.get("is_correct"):
                    print("✓ Correct answer!")
                else:
                    print(f"✗ Incorrect. Expected: {test_env.current_task.ground_truth}")
                break


if __name__ == "__main__":
    main()
