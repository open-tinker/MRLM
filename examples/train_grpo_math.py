"""
Example: Training a language model with GRPO on math reasoning tasks.

GRPO (Group Relative Policy Optimization) normalizes rewards within groups
to reduce variance. This example shows how to use GRPO for math problem solving.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mrlm.core.types import EnvironmentMode
from mrlm.core.llm_environment import LLMEnvironment
from mrlm.environments.math import MathReasoningEnvironment, MathProblemGenerator
from mrlm.algorithms.grpo import GRPOTrainer
from mrlm.config.training_config import (
    ExperimentConfig,
    TrainingConfig,
    ModelConfig,
    GRPOConfig,
    EnvironmentConfig,
)


def main():
    """Train LLM with GRPO on math reasoning."""

    # Configuration
    config = ExperimentConfig(
        experiment_name="grpo_math_reasoning",
        training=TrainingConfig(
            algorithm="grpo",
            num_epochs=50,
            batch_size=8,
            learning_rate=1e-5,
            max_episode_length=5,
            episodes_per_iteration=32,
        ),
        model=ModelConfig(
            model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
            device_map="auto",
            torch_dtype="float16",
        ),
        grpo=GRPOConfig(
            clip_range=0.2,
            gamma=1.0,  # No discounting for single-turn math
            gae_lambda=0.95,
            num_grpo_epochs=4,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            group_size=4,  # Generate 4 responses per prompt
            target_kl=0.01,
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

    # Create evaluation environments (math problems)
    print("Creating math reasoning environments...")
    problem_generator = MathProblemGenerator(difficulty_range=(1, 3))

    eval_envs = [
        MathReasoningEnvironment(
            problem_generator=problem_generator,
            mode=EnvironmentMode.CLIENT,
            max_turns=config.training.max_episode_length,
        )
        for _ in range(4)  # 4 parallel environments
    ]

    # Create GRPO trainer
    print("Initializing GRPO trainer...")
    trainer = GRPOTrainer(
        policy_env=policy_env,
        eval_envs=eval_envs,
        config=config,
        device=device,
    )

    # Train
    print(f"Starting GRPO training for {config.training.num_epochs} epochs...")
    print(f"Group size: {config.grpo.group_size} responses per prompt")
    print(f"Episodes per iteration: {config.training.episodes_per_iteration}")

    trainer.train(
        num_iterations=config.training.num_epochs,
        eval_every=5,
        save_every=10,
    )

    print("Training complete!")


if __name__ == "__main__":
    main()
