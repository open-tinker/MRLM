"""
Example: SFT (Supervised Fine-Tuning) for World Model Training.

This example demonstrates training an LLM to predict future states
from environment trajectories, functioning as a world model.

SFT can be used for:
1. Behavioral cloning: Learn to imitate expert actions
2. World model: Learn to predict next observations
3. Pre-training before RL fine-tuning
4. Regularization alongside RL training

The trainer collects trajectories from environments and trains the model
to predict either:
- Actions given observations (behavioral cloning)
- Next observations given current state and action (world model)
- Both (combined mode)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mrlm.core.types import EnvironmentMode
from mrlm.core.llm_environment import LLMEnvironment
from mrlm.environments.math import MathReasoningEnvironment, MathProblemGenerator
from mrlm.algorithms.sft import SFTTrainer, TrajectoryDataset
from mrlm.config.training_config import (
    ExperimentConfig,
    TrainingConfig,
    ModelConfig,
    SFTConfig,
)


def main():
    """Train LLM with SFT for world model learning."""

    # Configuration
    config = ExperimentConfig(
        experiment_name="sft_world_model_math",
        description="Train LLM to predict future states from math problem trajectories",
        training=TrainingConfig(
            algorithm="sft",
            num_epochs=50,
            batch_size=8,
            learning_rate=5e-6,
            weight_decay=0.01,
            max_episode_length=5,
            episodes_per_iteration=16,  # Collect 16 trajectories per iteration
        ),
        model=ModelConfig(
            model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
            device_map="auto",
            torch_dtype="float16",
        ),
        sft=SFTConfig(
            mode="combined",  # Train both BC and world model
            bc_weight=0.5,  # Weight for behavioral cloning loss
            world_model_weight=0.5,  # Weight for world model loss
            filter_low_reward=True,  # Only keep successful trajectories
            min_reward_threshold=0.5,  # Minimum reward to keep
            collect_every=5,  # Collect new trajectories every 5 iterations
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

    # Optional: Load pre-collected trajectories
    # trajectory_dataset = TrajectoryDataset.load("path/to/trajectories.json")
    trajectory_dataset = None

    # Create SFT trainer
    print("Initializing SFT trainer...")
    trainer = SFTTrainer(
        policy_env=policy_env,
        eval_envs=eval_envs,
        config=config,
        trajectory_dataset=trajectory_dataset,
        device=device,
    )

    # Train
    print(f"\nStarting SFT training for {config.training.num_epochs} epochs...")
    print(f"Mode: {config.sft.mode}")
    print(f"Behavioral cloning weight: {config.sft.bc_weight}")
    print(f"World model weight: {config.sft.world_model_weight}")
    print()
    print("The model will learn to:")
    if config.sft.mode in ["behavioral_cloning", "combined"]:
        print("  ✓ Predict actions given observations (behavioral cloning)")
    if config.sft.mode in ["world_model", "combined"]:
        print("  ✓ Predict next states given current state and action (world model)")
    print()

    trainer.train(
        num_iterations=config.training.num_epochs,
        collect_every=config.sft.collect_every,
        eval_every=5,
        save_every=10,
    )

    print("\nTraining complete!")
    print(f"Final dataset size: {len(trainer.trajectory_dataset)} trajectories")
    print(f"Total transitions: {trainer.trajectory_dataset.num_transitions()}")


if __name__ == "__main__":
    main()
