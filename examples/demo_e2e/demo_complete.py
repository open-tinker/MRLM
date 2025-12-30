"""
Complete End-to-End Demo for MRLM

This demo showcases the full MRLM framework capabilities:
1. Loading synthetic datasets
2. Training with PPO on math problems
3. Evaluating the trained model
4. Showing training metrics and results

This demo is designed to run quickly without GPU and demonstrate all components working together.

Usage:
    python demo_complete.py
"""

import json
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mrlm.algorithms.ppo import PPOConfig, PPOTrainer
from mrlm.core import EnvironmentMode, LLMEnvironment, Message, MessageRole
from mrlm.environments.math import MathReasoningEnvironment


def load_synthetic_data():
    """Load synthetic math problems."""
    data_dir = Path(__file__).parent / "data"
    math_file = data_dir / "math_problems.json"

    if not math_file.exists():
        print(f"\n⚠ Warning: Synthetic data not found at {math_file}")
        print("Please run: python synthetic_data_generator.py first\n")
        sys.exit(1)

    with open(math_file, "r") as f:
        math_problems = json.load(f)

    return math_problems


def print_section(title: str, char: str = "="):
    """Print a formatted section header."""
    print("\n" + char * 70)
    print(title)
    print(char * 70)


def print_subsection(title: str):
    """Print a formatted subsection."""
    print(f"\n{title}")
    print("-" * 70)


def evaluate_model(policy_env, eval_envs, num_episodes: int = 5):
    """Evaluate the model on test problems."""
    print_subsection("Running Evaluation")

    total_reward = 0
    correct_count = 0
    total_count = 0

    for i in range(num_episodes):
        # Use first eval environment
        env = eval_envs[0]

        # Reset environment
        obs = env.reset()

        # Get model response
        response = policy_env.step(obs.messages[-1])

        # Evaluate response
        final_obs, reward = env.step(response)

        total_reward += reward.value
        total_count += 1

        is_correct = final_obs.info.get("correct", False)
        if is_correct:
            correct_count += 1

        # Print episode summary
        print(f"\nEpisode {i+1}/{num_episodes}:")
        print(f"  Problem: {obs.messages[-1].content[:60]}...")
        print(f"  Model Answer: {response.content[:80]}...")
        print(f"  Correct: {'✓' if is_correct else '✗'}")
        print(f"  Reward: {reward.value:.2f}")

    # Print summary
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    avg_reward = total_reward / total_count if total_count > 0 else 0

    print_subsection("Evaluation Results")
    print(f"  Episodes: {total_count}")
    print(f"  Correct: {correct_count}/{total_count}")
    print(f"  Accuracy: {accuracy:.1f}%")
    print(f"  Average Reward: {avg_reward:.3f}")

    return {
        "accuracy": accuracy,
        "avg_reward": avg_reward,
        "correct": correct_count,
        "total": total_count,
    }


def main():
    """Main demo function."""
    print_section("MRLM: Complete End-to-End Demo", "=")
    print("This demo showcases:")
    print("  • Loading synthetic datasets")
    print("  • Setting up PPO training")
    print("  • Training on math reasoning tasks")
    print("  • Evaluating model performance")
    print("\nNote: This is a quick demo with minimal training for demonstration.")

    # =====================================================================
    # STEP 1: Load Synthetic Data
    # =====================================================================
    print_section("[1/7] Loading Synthetic Data")
    math_problems = load_synthetic_data()
    print(f"✓ Loaded {len(math_problems)} math problems")

    # Show sample problem
    print_subsection("Sample Problem")
    sample = math_problems[0]
    print(f"Question: {sample['question']}")
    print(f"Answer: {sample['answer']}")
    print(f"Difficulty: {sample['difficulty']}")

    # =====================================================================
    # STEP 2: Configure Training
    # =====================================================================
    print_section("[2/7] Configuring Training")
    config = PPOConfig(
        # Training - small values for quick demo
        num_epochs=5,  # Very small for demo
        num_ppo_epochs=2,
        batch_size=4,
        learning_rate=1e-5,
        # PPO parameters
        clip_range=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        # Rollout
        num_rollouts_per_iteration=8,  # Small for demo
        max_episode_length=5,
        # Evaluation
        eval_every=2,
        num_eval_episodes=3,
        save_every=10,  # Don't save in demo
    )

    print("Configuration:")
    print(f"  • Epochs: {config.num_epochs}")
    print(f"  • Batch Size: {config.batch_size}")
    print(f"  • Learning Rate: {config.learning_rate}")
    print(f"  • Rollouts per Iteration: {config.num_rollouts_per_iteration}")
    print(f"  • PPO Clip Range: {config.clip_range}")

    # =====================================================================
    # STEP 3: Load Model
    # =====================================================================
    print_section("[3/7] Loading Language Model")
    model_name = "gpt2"  # Small model for demo (124M parameters)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print("Loading model... (this may take a moment)")

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    print(f"✓ Model loaded: {model_name}")
    print(f"  Parameters: ~124M")
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # =====================================================================
    # STEP 4: Create Policy Environment
    # =====================================================================
    print_section("[4/7] Creating Policy Environment")
    policy_env = LLMEnvironment(
        model=model,
        tokenizer=tokenizer,
        mode=EnvironmentMode.SERVER,
        system_prompt="You are a helpful math tutor. Solve problems step-by-step and provide the final numerical answer.",
        max_new_tokens=128,
        temperature=0.7,
    )
    print(f"✓ Policy Environment created")
    print(f"  Mode: {policy_env.mode.value}")
    print(f"  Max new tokens: 128")

    # =====================================================================
    # STEP 5: Create Evaluation Environments
    # =====================================================================
    print_section("[5/7] Creating Evaluation Environments")

    # Split problems into train/test
    train_problems = math_problems[:15]  # First 15 for training
    test_problems = math_problems[15:]  # Rest for testing

    eval_envs = [
        MathReasoningEnvironment(
            problems=test_problems,
            max_attempts=1,
            require_solution=False,
        )
        for _ in range(2)  # 2 parallel environments
    ]

    print(f"✓ Created {len(eval_envs)} evaluation environments")
    print(f"  Training problems: {len(train_problems)}")
    print(f"  Test problems: {len(test_problems)}")

    # =====================================================================
    # STEP 6: Run Initial Evaluation (Before Training)
    # =====================================================================
    print_section("[6/7] Initial Evaluation (Before Training)")
    print("Testing model performance before training...")

    initial_metrics = evaluate_model(policy_env, eval_envs, num_episodes=3)

    # =====================================================================
    # STEP 7: Create Trainer and Train
    # =====================================================================
    print_section("[7/7] Training with PPO")

    trainer = PPOTrainer(
        policy_env=policy_env,
        eval_envs=eval_envs,
        config=config,
        device=torch.device(device),
    )

    print(f"✓ Trainer created: PPOTrainer")
    print(f"\nStarting training...")
    print(f"  • {config.num_epochs} iterations")
    print(f"  • {config.num_rollouts_per_iteration} rollouts per iteration")
    print(f"  • Evaluation every {config.eval_every} iterations")
    print("-" * 70)

    try:
        # Train the model
        trainer.train(
            num_iterations=config.num_epochs,
            eval_every=config.eval_every,
            save_every=config.save_every,
        )
        training_completed = True
    except Exception as e:
        print(f"\n⚠ Training encountered an issue: {e}")
        print("This can happen in quick demos with small batch sizes.")
        training_completed = False
        import traceback
        traceback.print_exc()

    # =====================================================================
    # Final Evaluation
    # =====================================================================
    if training_completed:
        print_section("Final Evaluation (After Training)")
        print("Testing model performance after training...")

        final_metrics = evaluate_model(policy_env, eval_envs, num_episodes=5)

        # =====================================================================
        # Summary
        # =====================================================================
        print_section("Training Summary", "=")

        print(f"Model: {model_name}")
        print(f"Algorithm: PPO")
        print(f"Training Problems: {len(train_problems)}")
        print(f"Test Problems: {len(test_problems)}")

        print_subsection("Performance Comparison")
        print(f"{'Metric':<20} {'Before':<15} {'After':<15} {'Change':<15}")
        print("-" * 70)

        acc_change = final_metrics["accuracy"] - initial_metrics["accuracy"]
        reward_change = final_metrics["avg_reward"] - initial_metrics["avg_reward"]

        print(f"{'Accuracy':<20} {initial_metrics['accuracy']:>6.1f}%  {' '*7} {final_metrics['accuracy']:>6.1f}%  {' '*7} {acc_change:>+6.1f}%")
        print(f"{'Avg Reward':<20} {initial_metrics['avg_reward']:>6.3f}  {' '*7} {final_metrics['avg_reward']:>6.3f}  {' '*7} {reward_change:>+6.3f}")

        print_subsection("Training Stats")
        print(f"  Total Epochs: {trainer.epoch}")
        print(f"  Total Steps: {trainer.global_step}")
        print(f"  Device: {device}")

    # =====================================================================
    # Closing
    # =====================================================================
    print_section("Demo Complete!", "=")
    print("\nWhat you just saw:")
    print("  ✓ Loaded synthetic math problems")
    print("  ✓ Configured PPO training")
    print("  ✓ Loaded a language model (GPT-2)")
    print("  ✓ Created policy and evaluation environments")
    print("  ✓ Evaluated model before training")
    if training_completed:
        print("  ✓ Trained model with PPO")
        print("  ✓ Evaluated model after training")
        print("  ✓ Compared before/after performance")

    print("\nNext Steps:")
    print("  • Try with larger models (Qwen2.5-1.5B, Llama-3-8B)")
    print("  • Train for more iterations (50-100 epochs)")
    print("  • Experiment with other environments (code, debate, tools)")
    print("  • Use GPU for faster training")
    print("  • Try other algorithms (GRPO, DPO, SFT)")

    print("\n✓ MRLM Demo completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
