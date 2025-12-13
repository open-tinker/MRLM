"""
Train from YAML Configuration

This script demonstrates training MRLM models using YAML configuration files.
This is the recommended way to configure training runs.

Usage:
    python train_from_config.py --config configs/code_generation_ppo.yaml
    python train_from_config.py --config configs/math_reasoning_ppo.yaml
    python train_from_config.py --config configs/quick_start.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mrlm.algorithms.ppo import PPOConfig, PPOTrainer
from mrlm.config import load_config
from mrlm.core import EnvironmentMode, LLMEnvironment
from mrlm.environments import CodeExecutionEnvironment, MathReasoningEnvironment

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_environment(env_config):
    """
    Create environment from configuration.

    Args:
        env_config: EnvironmentConfig object

    Returns:
        Environment instance
    """
    env_type = env_config.env_type.lower()

    if env_type == "code":
        return CodeExecutionEnvironment(
            dataset=env_config.dataset,
            timeout=env_config.timeout,
            max_attempts=env_config.max_attempts,
        )
    elif env_type == "math":
        return MathReasoningEnvironment(
            dataset=env_config.dataset, max_attempts=env_config.max_attempts
        )
    else:
        raise ValueError(f"Unknown environment type: {env_type}")


def main():
    parser = argparse.ArgumentParser(description="Train MRLM from YAML configuration")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--output-dir", type=str, help="Override output directory from config"
    )
    parser.add_argument("--device", type=str, help="Override device (cuda, cpu, auto)")
    args = parser.parse_args()

    print("=" * 70)
    print("MRLM: Training from Configuration")
    print("=" * 70)

    # Load configuration
    print(f"\n[1/7] Loading configuration from {args.config}...")
    config = load_config(args.config)

    # Override if specified
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.device:
        config.model.device = args.device

    # Validate
    issues = config.validate()
    if issues:
        print("\nConfiguration validation:")
        for issue in issues:
            print(f"  ⚠ {issue}")

    print(f"\n✓ Configuration loaded")
    print(f"  Experiment: {config.experiment_name}")
    print(f"  Algorithm: {config.training.algorithm.upper()}")
    print(f"  Model: {config.model.model_name_or_path}")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Output: {config.output_dir}")

    # Load model
    print(f"\n[2/7] Loading model...")
    device = config.model.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name_or_path,
        torch_dtype=torch.bfloat16 if config.training.use_bf16 else None,
        trust_remote_code=config.model.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_name_or_path or config.model.model_name_or_path
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  ✓ {config.model.model_name_or_path} loaded")
    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # Create policy environment
    print(f"\n[3/7] Creating policy environment...")
    from mrlm.core.types import GenerationConfig

    gen_config = GenerationConfig(
        max_new_tokens=config.model.max_new_tokens,
        temperature=config.model.temperature,
        top_p=config.model.top_p,
        top_k=config.model.top_k,
        do_sample=config.model.do_sample,
    )

    policy_env = LLMEnvironment(
        model=model,
        tokenizer=tokenizer,
        mode=EnvironmentMode.SERVER,
        generation_config=gen_config,
    )
    print(f"  ✓ Policy environment created")

    # Create evaluation environments
    print(f"\n[4/7] Creating evaluation environments...")
    eval_envs = []
    for env_config in config.eval_envs:
        env = create_environment(env_config)
        eval_envs.append(env)

    print(f"  ✓ Created {len(eval_envs)} evaluation environments")
    for i, env in enumerate(eval_envs):
        print(f"    [{i+1}] {env}")

    # Get algorithm config
    print(f"\n[5/7] Configuring {config.training.algorithm.upper()}...")
    algo_config = config.get_algorithm_config()

    if config.training.algorithm.lower() == "ppo":
        # Convert to PPOConfig expected by trainer
        from mrlm.algorithms.ppo.config import PPOConfig as TrainerPPOConfig

        trainer_config = TrainerPPOConfig(
            num_epochs=config.training.num_epochs,
            num_ppo_epochs=algo_config.num_ppo_epochs,
            batch_size=config.training.batch_size,
            mini_batch_size=config.training.mini_batch_size or config.training.batch_size,
            learning_rate=config.training.learning_rate,
            max_grad_norm=config.training.max_grad_norm,
            clip_range=algo_config.clip_range,
            gamma=algo_config.gamma,
            gae_lambda=algo_config.gae_lambda,
            value_loss_coef=algo_config.value_loss_coef,
            entropy_coef=algo_config.entropy_coef,
            value_clip_range=algo_config.clip_range_vf,
            num_rollouts_per_iteration=config.training.num_rollouts_per_iteration,
            max_episode_length=config.training.max_episode_length,
            eval_every=config.training.eval_every,
            eval_episodes=config.training.eval_episodes,
            save_every=config.training.save_every,
            log_every=config.training.log_every,
        )
    else:
        raise ValueError(f"Algorithm {config.training.algorithm} not yet supported")

    print(f"  ✓ {config.training.algorithm.upper()} configured")

    # Create trainer
    print(f"\n[6/7] Creating trainer...")
    trainer = PPOTrainer(
        policy_env=policy_env,
        eval_envs=eval_envs,
        config=trainer_config,
        device=torch.device(device),
        checkpoint_dir=Path(config.training.checkpoint_dir),
    )
    print(f"  ✓ {trainer}")

    # Train
    print(f"\n[7/7] Starting training...")
    print("  " + "-" * 60)
    print(f"  Experiment: {config.experiment_name}")
    print(f"  {config.description}")
    print("  " + "-" * 60)

    try:
        trainer.train(
            num_iterations=config.training.num_epochs,
            eval_every=config.training.eval_every,
            save_every=config.training.save_every,
        )

        # Save final model
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        print(f"\n✓ Training complete!")
        print(f"  Model saved to: {output_dir}")
        print(f"  Checkpoints: {config.training.checkpoint_dir}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n⚠ Error: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 70)
    print("Training session complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
