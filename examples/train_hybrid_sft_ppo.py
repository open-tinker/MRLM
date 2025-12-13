"""
Example: Hybrid Training Pipeline (SFT → PPO).

This example demonstrates a two-stage training approach:
1. Stage 1 (SFT): Pre-train with supervised fine-tuning on collected trajectories
2. Stage 2 (PPO): Fine-tune with reinforcement learning for optimization

This hybrid approach often yields better results than either method alone:
- SFT provides a good initialization
- PPO optimizes for the reward function
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

from mrlm.core.types import EnvironmentMode
from mrlm.core.llm_environment import LLMEnvironment
from mrlm.environments.math import MathReasoningEnvironment, MathProblemGenerator
from mrlm.algorithms.sft import SFTTrainer, TrajectoryDataset
from mrlm.algorithms.ppo import PPOTrainer
from mrlm.config.training_config import (
    ExperimentConfig,
    TrainingConfig,
    ModelConfig,
    SFTConfig,
    PPOConfig,
)


def stage1_sft_pretraining(model, tokenizer, eval_envs, device):
    """
    Stage 1: SFT pre-training.

    Learn from environment interactions to bootstrap the policy.
    """
    print("\n" + "=" * 70)
    print("STAGE 1: SFT PRE-TRAINING")
    print("=" * 70)

    # SFT configuration
    sft_config = ExperimentConfig(
        experiment_name="hybrid_sft_stage",
        description="SFT pre-training stage",
        training=TrainingConfig(
            algorithm="sft",
            num_epochs=30,
            batch_size=8,
            learning_rate=5e-6,
            max_episode_length=5,
            episodes_per_iteration=16,
        ),
        model=ModelConfig(
            model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",  # Not used (model already loaded)
        ),
        sft=SFTConfig(
            mode="combined",  # Train both BC and world model
            bc_weight=0.6,  # Focus more on behavioral cloning
            world_model_weight=0.4,
            filter_low_reward=True,
            min_reward_threshold=0.3,
            collect_every=3,
        ),
    )

    # Create policy environment
    policy_env = LLMEnvironment(
        model=model,
        tokenizer=tokenizer,
        mode=EnvironmentMode.SERVER,
        generation_config=sft_config.model.generation.to_generation_config(),
    )

    # Create SFT trainer
    print("\nInitializing SFT trainer...")
    sft_trainer = SFTTrainer(
        policy_env=policy_env,
        eval_envs=eval_envs,
        config=sft_config,
        device=device,
    )

    # Train with SFT
    print(f"\nPre-training with SFT for {sft_config.training.num_epochs} epochs...")
    print("Learning to:")
    print("  • Predict actions from observations (behavioral cloning)")
    print("  • Predict next states (world model)")
    print()

    sft_trainer.train(
        num_iterations=sft_config.training.num_epochs,
        collect_every=sft_config.sft.collect_every,
        eval_every=5,
        save_every=10,
    )

    # Save trajectory dataset for analysis
    dataset_path = Path("outputs/hybrid/sft_trajectories.json")
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    sft_trainer.trajectory_dataset.save(dataset_path)

    print(f"\n✅ SFT pre-training complete!")
    print(f"Collected {len(sft_trainer.trajectory_dataset)} trajectories")
    print(f"Dataset saved to: {dataset_path}")

    return model


def stage2_ppo_finetuning(model, tokenizer, eval_envs, device):
    """
    Stage 2: PPO fine-tuning.

    Optimize the pre-trained policy with reinforcement learning.
    """
    print("\n" + "=" * 70)
    print("STAGE 2: PPO FINE-TUNING")
    print("=" * 70)

    # PPO configuration
    ppo_config = ExperimentConfig(
        experiment_name="hybrid_ppo_stage",
        description="PPO fine-tuning stage",
        training=TrainingConfig(
            algorithm="ppo",
            num_epochs=50,
            batch_size=16,
            learning_rate=3e-6,  # Lower LR for fine-tuning
            max_episode_length=5,
            episodes_per_iteration=16,
        ),
        model=ModelConfig(
            model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",  # Not used
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

    # Create policy environment
    policy_env = LLMEnvironment(
        model=model,
        tokenizer=tokenizer,
        mode=EnvironmentMode.SERVER,
        generation_config=ppo_config.model.generation.to_generation_config(),
    )

    # Create PPO trainer
    print("\nInitializing PPO trainer...")
    ppo_trainer = PPOTrainer(
        policy_env=policy_env,
        eval_envs=eval_envs,
        config=ppo_config,
        device=device,
    )

    # Train with PPO
    print(f"\nFine-tuning with PPO for {ppo_config.training.num_epochs} epochs...")
    print("Optimizing for reward maximization\n")

    ppo_trainer.train(
        num_iterations=ppo_config.training.num_epochs,
        eval_every=10,
        save_every=10,
    )

    print(f"\n✅ PPO fine-tuning complete!")

    return model


def main():
    """
    Run hybrid training pipeline: SFT → PPO.
    """
    print("🚀 Hybrid Training Pipeline: SFT → PPO")
    print("=" * 70)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model and tokenizer
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create evaluation environments (shared across both stages)
    print("\nCreating evaluation environments...")
    problem_generator = MathProblemGenerator(difficulty_range=(1, 3))

    eval_envs = [
        MathReasoningEnvironment(
            problem_generator=problem_generator,
            mode=EnvironmentMode.CLIENT,
            max_turns=5,
        )
        for _ in range(4)
    ]
    print(f"Created {len(eval_envs)} math reasoning environments")

    # Stage 1: SFT Pre-training
    model = stage1_sft_pretraining(model, tokenizer, eval_envs, device)

    # Optional: Save intermediate checkpoint
    intermediate_path = Path("outputs/hybrid/after_sft")
    intermediate_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(intermediate_path)
    tokenizer.save_pretrained(intermediate_path)
    print(f"\nIntermediate checkpoint saved to: {intermediate_path}")

    # Stage 2: PPO Fine-tuning
    model = stage2_ppo_finetuning(model, tokenizer, eval_envs, device)

    # Save final model
    final_path = Path("outputs/hybrid/final")
    final_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    print("\n" + "=" * 70)
    print("✅ HYBRID TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nFinal model saved to: {final_path}")
    print("\nTraining Summary:")
    print("  Stage 1 (SFT):  Learned from demonstrations → good initialization")
    print("  Stage 2 (PPO):  Optimized for rewards → better performance")
    print("\nThis hybrid approach combines the best of both:")
    print("  ✓ SFT provides sample-efficient learning from trajectories")
    print("  ✓ PPO refines the policy for reward maximization")


if __name__ == "__main__":
    main()
