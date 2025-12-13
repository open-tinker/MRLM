"""
Example: Multi-Environment Training with PPO.

This example demonstrates training a single model on multiple environments
simultaneously. The model learns to generalize across different task types.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mrlm.core.types import EnvironmentMode
from mrlm.core.llm_environment import LLMEnvironment
from mrlm.environments.code import CodeExecutionEnvironment, CodeProblemGenerator
from mrlm.environments.math import MathReasoningEnvironment, MathProblemGenerator
from mrlm.environments.tools import ToolUseEnvironment
from mrlm.environments.tools.builtin_tools import create_default_tool_registry
from mrlm.algorithms.ppo import PPOTrainer
from mrlm.config.training_config import (
    ExperimentConfig,
    TrainingConfig,
    ModelConfig,
    PPOConfig,
)


def main():
    """Train on multiple environments simultaneously."""

    # Configuration
    config = ExperimentConfig(
        experiment_name="multi_env_ppo",
        description="Train on code, math, and tool-use environments simultaneously",
        training=TrainingConfig(
            algorithm="ppo",
            num_epochs=100,
            batch_size=16,
            learning_rate=5e-6,
            max_episode_length=5,
            episodes_per_iteration=24,  # 8 per environment type
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

    # Create multiple environment types
    print("\nCreating diverse evaluation environments...")

    # Code execution environments
    code_generator = CodeProblemGenerator()
    code_envs = [
        CodeExecutionEnvironment(
            problem_generator=code_generator,
            mode=EnvironmentMode.CLIENT,
            max_turns=3,
        )
        for _ in range(2)
    ]
    print(f"  ✓ {len(code_envs)} Code execution environments")

    # Math reasoning environments
    math_generator = MathProblemGenerator(difficulty_range=(1, 3))
    math_envs = [
        MathReasoningEnvironment(
            problem_generator=math_generator,
            mode=EnvironmentMode.CLIENT,
            max_turns=5,
        )
        for _ in range(2)
    ]
    print(f"  ✓ {len(math_envs)} Math reasoning environments")

    # Tool use environments
    tool_registry = create_default_tool_registry()
    tool_envs = [
        ToolUseEnvironment(
            tool_registry=tool_registry,
            mode=EnvironmentMode.CLIENT,
            max_turns=5,
        )
        for _ in range(2)
    ]
    print(f"  ✓ {len(tool_envs)} Tool use environments")

    # Combine all environments
    eval_envs = code_envs + math_envs + tool_envs
    print(f"\nTotal environments: {len(eval_envs)}")

    # Create PPO trainer
    print("\nInitializing PPO trainer...")
    trainer = PPOTrainer(
        policy_env=policy_env,
        eval_envs=eval_envs,
        config=config,
        device=device,
    )

    # Train
    print(f"\n{'=' * 70}")
    print(f"Starting Multi-Environment Training")
    print(f"{'=' * 70}")
    print(f"Model will learn to solve:")
    print(f"  • Code generation and execution")
    print(f"  • Mathematical reasoning")
    print(f"  • Tool use and planning")
    print(f"\nTraining for {config.training.num_epochs} epochs...")
    print(f"{'=' * 70}\n")

    trainer.train(
        num_iterations=config.training.num_epochs,
        eval_every=10,
        save_every=20,
    )

    print("\n✅ Multi-environment training complete!")
    print("The model is now a generalist capable of:")
    print("  ✓ Writing and debugging code")
    print("  ✓ Solving math problems")
    print("  ✓ Using tools to answer questions")


if __name__ == "__main__":
    main()
