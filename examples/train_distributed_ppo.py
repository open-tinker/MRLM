"""
Example: Distributed PPO training with FSDP or DDP.

This example shows how to train with multiple GPUs using either:
- FSDP (Fully Sharded Data Parallel) for large models
- DDP (Distributed Data Parallel) for smaller models

Launch with torchrun:
    # Single node, 4 GPUs with FSDP:
    torchrun --nproc_per_node=4 train_distributed_ppo.py --strategy fsdp

    # Single node, 4 GPUs with DDP:
    torchrun --nproc_per_node=4 train_distributed_ppo.py --strategy ddp

    # Multi-node (2 nodes, 4 GPUs each):
    # Node 0:
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
        --master_addr=<node0_ip> --master_port=12355 \
        train_distributed_ppo.py --strategy fsdp

    # Node 1:
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
        --master_addr=<node0_ip> --master_port=12355 \
        train_distributed_ppo.py --strategy fsdp
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mrlm.core.types import EnvironmentMode
from mrlm.core.llm_environment import LLMEnvironment
from mrlm.environments.code import CodeExecutionEnvironment, CodeProblemGenerator
from mrlm.algorithms.ppo import PPOTrainer
from mrlm.config.training_config import (
    ExperimentConfig,
    TrainingConfig,
    ModelConfig,
    PPOConfig,
    DistributedConfig,
)
from mrlm.distributed import (
    init_distributed,
    cleanup_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    setup_for_distributed_training,
    get_device,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed PPO Training")
    parser.add_argument(
        "--strategy",
        type=str,
        default="ddp",
        choices=["ddp", "fsdp", "none"],
        help="Distributed strategy",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Learning rate",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize distributed training
    init_distributed(backend="nccl")

    rank = get_rank()
    world_size = get_world_size()
    device = get_device()

    if is_main_process():
        print(f"Distributed training initialized: {world_size} processes")
        print(f"Strategy: {args.strategy}")

    # Configuration
    config = ExperimentConfig(
        experiment_name=f"distributed_ppo_{args.strategy}",
        training=TrainingConfig(
            algorithm="ppo",
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_episode_length=3,
            episodes_per_iteration=world_size * 8,  # Scale with world size
        ),
        model=ModelConfig(
            model_name_or_path=args.model,
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
        distributed=DistributedConfig(
            enabled=True,
            strategy=args.strategy,
            backend="nccl",
        ),
    )

    # Load model and tokenizer
    if is_main_process():
        print(f"Loading model: {args.model}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
    )
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Setup distributed model
    if args.strategy != "none":
        if is_main_process():
            print(f"Setting up {args.strategy.upper()}...")

        model = setup_for_distributed_training(
            model,
            strategy=args.strategy,
            sharding_strategy="full_shard" if args.strategy == "fsdp" else None,
            mixed_precision=True,
        )

    # Create policy environment
    policy_env = LLMEnvironment(
        model=model,
        tokenizer=tokenizer,
        mode=EnvironmentMode.SERVER,
        generation_config=config.model.generation.to_generation_config(),
    )

    # Create evaluation environments (one per rank)
    problem_generator = CodeProblemGenerator()
    eval_envs = [
        CodeExecutionEnvironment(
            problem_generator=problem_generator,
            mode=EnvironmentMode.CLIENT,
            max_turns=config.training.max_episode_length,
        )
    ]

    # Create trainer
    if is_main_process():
        print("Initializing PPO trainer...")

    trainer = PPOTrainer(
        policy_env=policy_env,
        eval_envs=eval_envs,
        config=config,
        device=device,
    )

    # Train
    if is_main_process():
        print(f"Starting distributed training for {args.num_epochs} epochs...")
        print(f"Effective batch size: {args.batch_size * world_size}")

    try:
        trainer.train(
            num_iterations=args.num_epochs,
            eval_every=5,
            save_every=10,
        )

        if is_main_process():
            print("Training complete!")

    finally:
        # Cleanup
        cleanup_distributed()


if __name__ == "__main__":
    main()
