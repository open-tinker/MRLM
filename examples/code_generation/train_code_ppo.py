"""
PPO Training for Code Generation

This example demonstrates training an LLM on code generation tasks
using PPO with the CodeExecutionEnvironment.

This shows realistic LLM RL training on programming problems.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mrlm.algorithms.ppo import PPOConfig, PPOTrainer
from mrlm.core import EnvironmentMode, LLMEnvironment
from mrlm.environments.code import CodeExecutionEnvironment


def main():
    print("=" * 70)
    print("MRLM: PPO Training for Code Generation")
    print("=" * 70)

    # Configuration
    print("\n[1/6] Configuration...")
    config = PPOConfig(
        num_epochs=20,  # Training iterations
        num_ppo_epochs=3,
        batch_size=16,
        learning_rate=5e-6,  # Lower LR for stability
        clip_range=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        num_rollouts_per_iteration=32,
        max_episode_length=50,
        eval_every=5,
        save_every=5,
    )
    print(f"   ✓ Training for {config.num_epochs} iterations")

    # Model
    print("\n[2/6] Loading model...")
    model_name = "gpt2"  # Can use larger models: "Qwen/Qwen2.5-1.5B", etc.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"   Loading {model_name} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"   ✓ Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters)")

    # Policy environment
    print("\n[3/6] Creating policy environment...")
    policy_env = LLMEnvironment(
        model=model,
        tokenizer=tokenizer,
        mode=EnvironmentMode.SERVER,
        system_prompt="You are an expert Python programmer.",
    )
    print(f"   ✓ {policy_env}")

    # Code execution environments
    print("\n[4/6] Creating code execution environments...")

    # Define custom problems (or use dataset="humaneval")
    custom_problems = [
        {
            "id": "sum_list",
            "prompt": "Write a function called 'sum_list' that takes a list of numbers and returns their sum.",
            "function_name": "sum_list",
            "test_cases": [
                (([1, 2, 3],), 6),
                (([],), 0),
                (([-1, 1],), 0),
                (([10, 20, 30],), 60),
            ],
        },
        {
            "id": "max_value",
            "prompt": "Write a function called 'max_value' that takes a list of numbers and returns the maximum value.",
            "function_name": "max_value",
            "test_cases": [
                (([1, 5, 3],), 5),
                (([10],), 10),
                (([-5, -1, -10],), -1),
            ],
        },
        {
            "id": "is_palindrome",
            "prompt": "Write a function called 'is_palindrome' that checks if a string is a palindrome (reads the same forwards and backwards).",
            "function_name": "is_palindrome",
            "test_cases": [
                (("racecar",), True),
                (("hello",), False),
                (("",), True),
                (("a",), True),
            ],
        },
    ]

    eval_envs = [
        CodeExecutionEnvironment(problems=custom_problems, timeout=3.0),
        CodeExecutionEnvironment(problems=custom_problems, timeout=3.0),
    ]

    print(f"   ✓ Created {len(eval_envs)} code execution environments")
    print(f"   ✓ {len(custom_problems)} programming problems loaded")

    # Trainer
    print("\n[5/6] Creating PPO trainer...")
    trainer = PPOTrainer(
        policy_env=policy_env,
        eval_envs=eval_envs,
        config=config,
        device=torch.device(device),
    )
    print(f"   ✓ {trainer}")

    # Train
    print("\n[6/6] Starting training...")
    print("   " + "-" * 60)
    print("   Training LLM to write Python code using PPO")
    print("   Rewards based on test case pass rate")
    print("   " + "-" * 60)

    try:
        trainer.train(
            num_iterations=config.num_epochs,
            eval_every=config.eval_every,
            save_every=config.save_every,
        )

        # Save final model
        output_dir = Path("./output/code_generation_ppo")
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        print(f"\n✓ Training complete! Model saved to {output_dir}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Partial progress has been saved.")

    except Exception as e:
        print(f"\n⚠ Training encountered an issue: {e}")
        print("Note: Some components use placeholder implementations.")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 70)
    print("Training session complete!")
    print("=" * 70)

    print("\nNext steps:")
    print("  - Try with larger models (Qwen2.5-7B, Llama-3-8B)")
    print("  - Use real datasets (HumanEval, MBPP)")
    print("  - Increase training iterations")
    print("  - Experiment with reward shaping")


if __name__ == "__main__":
    main()
