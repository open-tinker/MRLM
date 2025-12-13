"""
Training configuration for MRLM.

This module provides configuration classes for all aspects of RL training.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union


@dataclass
class TrainingConfig:
    """
    Main training configuration.

    This configuration controls the overall training process including
    the algorithm, hyperparameters, and training schedule.
    """

    # Algorithm
    algorithm: str = "ppo"  # ppo, grpo, dpo

    # Training schedule
    num_epochs: int = 100
    num_iterations: Optional[int] = None  # Alternative to num_epochs
    batch_size: int = 32
    mini_batch_size: Optional[int] = None
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5

    # Optimizer
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # Learning rate schedule
    lr_scheduler: Optional[str] = None  # constant, linear, cosine
    warmup_steps: int = 0

    # Rollout collection
    num_rollouts_per_iteration: int = 128
    max_episode_length: int = 512

    # Evaluation
    eval_every: int = 10
    eval_episodes: int = 10

    # Checkpointing
    save_every: int = 10
    checkpoint_dir: str = "./checkpoints"
    save_total_limit: Optional[int] = None  # Keep only N most recent checkpoints

    # Logging
    log_every: int = 1
    log_dir: str = "./logs"

    # Experiment tracking
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    use_tensorboard: bool = False

    # Reproducibility
    seed: Optional[int] = None

    # Mixed precision
    use_fp16: bool = False
    use_bf16: bool = False

    # Gradient accumulation
    gradient_accumulation_steps: int = 1


@dataclass
class PPOConfig:
    """PPO-specific hyperparameters."""

    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None  # Value function clipping
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_ppo_epochs: int = 4
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    target_kl: Optional[float] = None  # Early stopping based on KL
    normalize_advantages: bool = True


@dataclass
class GRPOConfig:
    """GRPO-specific hyperparameters."""

    group_size: int = 4  # Number of responses per prompt
    gamma: float = 0.99
    normalize_rewards: bool = True
    reward_normalization: str = "group"  # group, batch, running
    temperature: float = 1.0


@dataclass
class DPOConfig:
    """DPO-specific hyperparameters."""

    beta: float = 0.1  # Temperature parameter
    reference_free: bool = False  # Whether to use reference model
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"  # sigmoid, hinge, ipo


@dataclass
class SFTConfig:
    """SFT (Supervised Fine-Tuning) hyperparameters for world model training."""

    # Training mode
    mode: str = "combined"  # behavioral_cloning, world_model, combined

    # Loss weights (for combined mode)
    bc_weight: float = 0.5  # Behavioral cloning weight
    world_model_weight: float = 0.5  # World model weight

    # Trajectory filtering
    filter_low_reward: bool = False  # Filter out low-reward trajectories
    min_reward_threshold: float = 0.0  # Minimum reward to keep trajectory

    # Collection
    collect_every: int = 1  # Collect new trajectories every N iterations
    max_trajectories: int = 1000  # Maximum trajectories to keep in dataset


@dataclass
class ModelConfig:
    """
    Model configuration.

    Specifies which model to load and how to configure it.
    """

    # Model loading
    model_name_or_path: str = "gpt2"
    model_type: str = "causal_lm"  # causal_lm, seq2seq
    tokenizer_name_or_path: Optional[str] = None
    trust_remote_code: bool = False

    # Model architecture
    use_flash_attention: bool = False
    gradient_checkpointing: bool = False

    # Quantization
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # LoRA/PEFT
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None

    # Generation config
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    num_beams: int = 1
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0

    # Device placement
    device: str = "auto"  # auto, cuda, cpu, cuda:0, etc.
    device_map: Optional[str] = None  # auto, balanced, etc.


@dataclass
class EnvironmentConfig:
    """
    Environment configuration.

    Specifies which environments to use and how to configure them.
    """

    # Environment type
    env_type: str  # code, math, debate, tool, custom

    # Environment mode
    mode: str = "client"  # server, client

    # Remote environment (if using gRPC)
    remote: bool = False
    server_address: Optional[str] = None

    # Environment-specific kwargs
    dataset: Optional[str] = None
    problems: Optional[List[Dict]] = None
    timeout: float = 5.0
    max_attempts: int = 3

    # Custom environment
    custom_class: Optional[str] = None  # Module path to custom environment
    custom_kwargs: Dict = field(default_factory=dict)

    # Reward shaping
    reward_scale: float = 1.0
    reward_clip: Optional[float] = None
    reward_bias: float = 0.0


@dataclass
class DistributedConfig:
    """
    Distributed training configuration.
    """

    # Distributed training
    enabled: bool = False
    backend: str = "nccl"  # nccl, gloo, mpi

    # Multi-GPU
    num_gpus: int = 1
    gpu_ids: Optional[List[int]] = None

    # Multi-node
    num_nodes: int = 1
    node_rank: int = 0
    master_addr: str = "localhost"
    master_port: int = 29500

    # Parallelism strategy
    strategy: str = "ddp"  # ddp, fsdp, deepspeed

    # FSDP config
    fsdp_sharding_strategy: str = "full_shard"  # full_shard, shard_grad_op, no_shard
    fsdp_cpu_offload: bool = False

    # DeepSpeed config
    deepspeed_config: Optional[Union[str, Dict]] = None
    zero_stage: int = 2


@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration.

    This is the top-level configuration that includes all sub-configurations.
    """

    # Experiment metadata
    experiment_name: str = "mrlm_experiment"
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    # Sub-configurations
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: Optional[PPOConfig] = None
    grpo: Optional[GRPOConfig] = None
    dpo: Optional[DPOConfig] = None
    sft: Optional[SFTConfig] = None

    # Environments
    policy_env: Optional[EnvironmentConfig] = None
    eval_envs: List[EnvironmentConfig] = field(default_factory=list)

    # Distributed
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

    # Output
    output_dir: str = "./output"

    def get_algorithm_config(self):
        """Get the algorithm-specific config based on training.algorithm."""
        algo = self.training.algorithm.lower()
        if algo == "ppo":
            return self.ppo or PPOConfig()
        elif algo == "grpo":
            return self.grpo or GRPOConfig()
        elif algo == "dpo":
            return self.dpo or DPOConfig()
        else:
            raise ValueError(f"Unknown algorithm: {algo}")

    def validate(self) -> List[str]:
        """
        Validate configuration and return list of warnings/errors.

        Returns:
            List of validation messages (empty if valid)
        """
        issues = []

        # Check algorithm config exists
        algo = self.training.algorithm.lower()
        if algo == "ppo" and self.ppo is None:
            issues.append("PPO selected but no PPO config provided (using defaults)")
        elif algo == "grpo" and self.grpo is None:
            issues.append("GRPO selected but no GRPO config provided (using defaults)")
        elif algo == "dpo" and self.dpo is None:
            issues.append("DPO selected but no DPO config provided (using defaults)")

        # Check environments
        if not self.eval_envs:
            issues.append("Warning: No evaluation environments specified")

        # Check distributed config
        if self.distributed.enabled and self.distributed.num_gpus < 2:
            issues.append("Distributed enabled but num_gpus < 2")

        # Check mixed precision
        if self.training.use_fp16 and self.training.use_bf16:
            issues.append("Error: Cannot use both fp16 and bf16")

        return issues
