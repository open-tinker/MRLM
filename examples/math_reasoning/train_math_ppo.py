"""
PPO Training for Math Reasoning

This example demonstrates training an LLM on mathematical reasoning tasks
using PPO with the MathReasoningEnvironment.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mrlm.algorithms.ppo import PPOConfig, PPOTrainer
from mrlm.core import EnvironmentMode, LLMEnvironment
from mrlm.environments.math import MathReasoningEnvironment


def main():
    print("=" * 70)
    print("MRLM: PPO Training for Math Reasoning")
    print("=" * 70)

    # Config
    print("\n[1/6] Configuration...")
    config = PPOConfig(
        num_epochs=30,
        num_ppo_epochs=3,
        batch_size=16,
        learning_rate=5e-6,
        num_rollouts_per_iteration=32,
        eval_every=5,
        save_every=5,
    )
    print(f"   ✓ Training for {config.num_epochs} iterations")

    # Model
    print("\n[2/6] Loading model...")
    model_name = "gpt2"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"   ✓ {model_name} loaded on {device}")

    # Policy environment
    print("\n[3/6] Creating policy environment...")
    policy_env = LLMEnvironment(
        model=model,
        tokenizer=tokenizer,
        mode=EnvironmentMode.SERVER,
        system_prompt="You are a helpful math tutor. Solve problems and give numerical answers.",
    )
    print(f"   ✓ {policy_env}")

    # Math environments
    print("\n[4/6] Creating math reasoning environments...")

    # Use default demo problems or load from dataset
    eval_envs = [
        MathReasoningEnvironment(dataset=None, max_attempts=2),  # Uses demo problems
        MathReasoningEnvironment(dataset=None, max_attempts=2),
    ]

    print(f"   ✓ Created {len(eval_envs)} math environments")

    # Trainer
    print("\n[5/6] Creating trainer...")
    trainer = PPOTrainer(
        policy_env=policy_env,
        eval_envs=eval_envs,
        config=config,
        device=torch.device(device),
    )
    print(f"   ✓ {trainer}")

    # Train
    print("\n[6/6] Training...")
    print("   " + "-" * 60)
    print("   Training LLM on math word problems using PPO")
    print("   Rewards based on answer correctness")
    print("   " + "-" * 60)

    try:
        trainer.train()

        # Save
        output_dir = Path("./output/math_reasoning_ppo")
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        print(f"\n✓ Complete! Model saved to {output_dir}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n⚠ Issue: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 70)
    print("Session complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
