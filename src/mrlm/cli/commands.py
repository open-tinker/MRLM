"""
CLI command implementations.

Each command has:
- add_X_parser(): Add command to subparser
- X_command(): Execute the command
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def add_train_parser(subparsers):
    """Add train command parser."""
    parser = subparsers.add_parser(
        "train",
        help="Train a model from configuration",
        description="Train an RL model using YAML configuration",
    )

    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Override output directory",
    )

    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from checkpoint",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    return parser


def train_command(args):
    """Execute train command."""
    from mrlm.config.parser import ConfigParser
    from mrlm.core.types import EnvironmentMode
    from mrlm.core.llm_environment import LLMEnvironment

    print(f"🚀 MRLM Training")
    print(f"Config: {args.config}")
    print()

    # Load configuration
    print("Loading configuration...")
    config = ConfigParser.load(args.config)

    # Override output directory if specified
    if args.output:
        config.output.save_dir = str(args.output)

    print(f"Experiment: {config.experiment_name}")
    print(f"Algorithm: {config.training.algorithm}")
    print(f"Model: {config.model.model_name_or_path}")
    print()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading model: {config.model.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name_or_path,
        device_map=config.model.device_map,
        torch_dtype=getattr(torch, config.model.torch_dtype) if config.model.torch_dtype else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create policy environment
    policy_env = LLMEnvironment(
        model=model,
        tokenizer=tokenizer,
        mode=EnvironmentMode.SERVER,
        generation_config=config.model.generation.to_generation_config(),
    )

    # Create evaluation environments
    print("Creating environments...")
    from mrlm.cli.utils import create_environments

    eval_envs = create_environments(config.eval_envs)
    print(f"Created {len(eval_envs)} evaluation environments")

    # Create trainer based on algorithm
    print(f"\nInitializing {config.training.algorithm.upper()} trainer...")

    if config.training.algorithm == "ppo":
        from mrlm.algorithms.ppo import PPOTrainer

        trainer = PPOTrainer(policy_env, eval_envs, config, device=device)

    elif config.training.algorithm == "grpo":
        from mrlm.algorithms.grpo import GRPOTrainer

        trainer = GRPOTrainer(policy_env, eval_envs, config, device=device)

    elif config.training.algorithm == "dpo":
        from mrlm.algorithms.dpo import DPOTrainer, PreferenceDataset

        # Load or create preference dataset
        dataset = PreferenceDataset()  # TODO: Load from config
        trainer = DPOTrainer(policy_env, dataset, config, device=device)

    elif config.training.algorithm == "sft":
        from mrlm.algorithms.sft import SFTTrainer

        trainer = SFTTrainer(policy_env, eval_envs, config, device=device)

    else:
        print(f"❌ Unknown algorithm: {config.training.algorithm}")
        sys.exit(1)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        # TODO: Implement checkpoint loading
        pass

    # Train
    print(f"\n{'=' * 60}")
    print(f"Starting training for {config.training.num_epochs} epochs")
    print(f"{'=' * 60}\n")

    trainer.train(
        num_iterations=config.training.num_epochs,
        eval_every=config.output.eval_interval,
        save_every=config.output.save_interval,
    )

    print(f"\n✅ Training complete!")
    print(f"Output directory: {config.output.save_dir}")


def add_serve_parser(subparsers):
    """Add serve command parser."""
    parser = subparsers.add_parser(
        "serve",
        help="Start an environment server",
        description="Start a gRPC server hosting environments",
    )

    parser.add_argument(
        "-e",
        "--environments",
        type=str,
        required=True,
        help="Comma-separated list of environments (e.g., 'code,math,debate')",
    )

    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=50051,
        help="Port to listen on (default: 50051)",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="[::]",
        help="Host to bind to (default: [::])",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    return parser


def serve_command(args):
    """Execute serve command."""
    from mrlm.server.environment_server import serve
    from mrlm.cli.utils import create_environment_by_name

    print(f"🌐 MRLM Environment Server")
    print(f"Port: {args.port}")
    print(f"Host: {args.host}")
    print()

    # Parse environment names
    env_names = [name.strip() for name in args.environments.split(",")]

    # Create environments
    print("Creating environments...")
    environments = {}

    for name in env_names:
        try:
            env = create_environment_by_name(name)
            environments[name] = env
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    if not environments:
        print("\n❌ No environments created. Exiting.")
        sys.exit(1)

    print(f"\nServing {len(environments)} environments:")
    for name in environments:
        print(f"  • {name}")

    # Start server
    print(f"\n🚀 Starting server on {args.host}:{args.port}")
    print("Press Ctrl+C to stop\n")

    server = serve(environments, port=args.port, host=args.host)
    server.wait_for_termination()


def add_eval_parser(subparsers):
    """Add eval command parser."""
    parser = subparsers.add_parser(
        "eval",
        help="Evaluate a trained model",
        description="Evaluate a trained model on environments",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Model name or path to checkpoint",
    )

    parser.add_argument(
        "-e",
        "--environment",
        type=str,
        required=True,
        help="Environment to evaluate on",
    )

    parser.add_argument(
        "-n",
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to run (default: 10)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Save results to JSON file",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    return parser


def eval_command(args):
    """Execute eval command."""
    from mrlm.core.types import EnvironmentMode, Message, MessageRole
    from mrlm.core.llm_environment import LLMEnvironment
    from mrlm.models.generation import generate_response
    from mrlm.cli.utils import create_environment_by_name

    print(f"🔍 MRLM Model Evaluation")
    print(f"Model: {args.model}")
    print(f"Environment: {args.environment}")
    print(f"Episodes: {args.num_episodes}")
    print()

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create environment
    print(f"Creating environment: {args.environment}")
    env = create_environment_by_name(args.environment)

    # Run evaluation
    print(f"\nRunning evaluation...")
    results = {
        "model": args.model,
        "environment": args.environment,
        "num_episodes": args.num_episodes,
        "episodes": [],
    }

    total_reward = 0.0

    for ep in range(args.num_episodes):
        print(f"\nEpisode {ep + 1}/{args.num_episodes}")

        obs = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False

        episode_data = {"episode": ep + 1, "steps": []}

        while not done and episode_length < 100:
            # Generate action
            with torch.no_grad():
                response = generate_response(
                    model=model,
                    tokenizer=tokenizer,
                    messages=obs.messages,
                    device=device,
                )

            action = Message(role=MessageRole.ASSISTANT, content=response)

            # Take step
            next_obs, reward = env.step(action)

            episode_reward += reward.value
            episode_length += 1
            done = next_obs.done

            # Record step
            episode_data["steps"].append({
                "step": episode_length,
                "reward": reward.value,
                "done": done,
            })

            if args.verbose:
                print(f"  Step {episode_length}: reward={reward.value:.3f}")

            obs = next_obs

        total_reward += episode_reward
        episode_data["total_reward"] = episode_reward
        episode_data["length"] = episode_length
        results["episodes"].append(episode_data)

        print(f"  Total reward: {episode_reward:.3f} ({episode_length} steps)")

    # Summary
    avg_reward = total_reward / args.num_episodes
    results["average_reward"] = avg_reward

    print(f"\n{'=' * 60}")
    print(f"Evaluation Complete")
    print(f"{'=' * 60}")
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Episodes: {args.num_episodes}")

    # Save results if specified
    if args.output:
        print(f"\nSaving results to {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


def add_collect_parser(subparsers):
    """Add collect command parser."""
    parser = subparsers.add_parser(
        "collect",
        help="Collect trajectories from environments",
        description="Collect trajectory data for SFT training",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Model name or path",
    )

    parser.add_argument(
        "-e",
        "--environment",
        type=str,
        required=True,
        help="Environment name",
    )

    parser.add_argument(
        "-n",
        "--num-episodes",
        type=int,
        default=100,
        help="Number of episodes to collect (default: 100)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output path for trajectory dataset",
    )

    parser.add_argument(
        "--filter-reward",
        type=float,
        help="Minimum reward threshold (filter out low-reward trajectories)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    return parser


def collect_command(args):
    """Execute collect command."""
    from mrlm.algorithms.sft import TrajectoryDataset
    from mrlm.core.types import Message, MessageRole
    from mrlm.models.generation import generate_response
    from mrlm.cli.utils import create_environment_by_name

    print(f"📊 MRLM Trajectory Collection")
    print(f"Model: {args.model}")
    print(f"Environment: {args.environment}")
    print(f"Episodes: {args.num_episodes}")
    print()

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create environment
    print(f"Creating environment: {args.environment}")
    env = create_environment_by_name(args.environment)

    # Create dataset
    dataset = TrajectoryDataset()

    # Define policy function
    def policy(obs):
        with torch.no_grad():
            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                messages=obs.messages,
                device=device,
            )
        return Message(role=MessageRole.ASSISTANT, content=response)

    # Collect trajectories
    print(f"\nCollecting {args.num_episodes} trajectories...")

    collected = dataset.collect_from_environment(
        env=env,
        policy=policy,
        num_trajectories=args.num_episodes,
        max_steps=100,
    )

    print(f"Collected {collected} trajectories")

    # Filter if threshold specified
    if args.filter_reward is not None:
        print(f"\nFiltering trajectories (min reward: {args.filter_reward})...")
        original_size = len(dataset)
        dataset = dataset.filter_by_reward(min_reward=args.filter_reward)
        print(f"Kept {len(dataset)}/{original_size} trajectories")

    # Save dataset
    print(f"\nSaving to {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.save(args.output)

    print(f"\n✅ Collection complete!")
    print(f"Total trajectories: {len(dataset)}")
    print(f"Total transitions: {dataset.num_transitions()}")


def add_info_parser(subparsers):
    """Add info command parser."""
    parser = subparsers.add_parser(
        "info",
        help="Show system information",
        description="Display system and library information",
    )

    return parser


def info_command(args):
    """Execute info command."""
    import platform
    import mrlm

    print(f"{'=' * 60}")
    print(f"MRLM System Information")
    print(f"{'=' * 60}\n")

    # Library info
    print("📦 Library:")
    print(f"  Version: 0.1.0")
    print(f"  Location: {Path(mrlm.__file__).parent}")
    print()

    # Python info
    print("🐍 Python:")
    print(f"  Version: {platform.python_version()}")
    print(f"  Implementation: {platform.python_implementation()}")
    print()

    # System info
    print("💻 System:")
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  Architecture: {platform.machine()}")
    print()

    # PyTorch info
    print("🔥 PyTorch:")
    print(f"  Version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  cuDNN version: {torch.backends.cudnn.version()}")
        print(f"  GPU count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    print()

    # Algorithms
    print("🤖 Algorithms:")
    print("  ✓ PPO (Proximal Policy Optimization)")
    print("  ✓ GRPO (Group Relative Policy Optimization)")
    print("  ✓ DPO (Direct Preference Optimization)")
    print("  ✓ SFT (Supervised Fine-Tuning)")
    print()

    # Environments
    print("🌍 Environments:")
    print("  ✓ Code Execution")
    print("  ✓ Math Reasoning")
    print("  ✓ Multi-Agent Debate")
    print("  ✓ Tool Use")
    print()

    # Features
    print("✨ Features:")
    print("  ✓ Distributed training (FSDP/DDP)")
    print("  ✓ gRPC server-client architecture")
    print("  ✓ YAML configuration system")
    print("  ✓ Trajectory collection and filtering")
    print()
