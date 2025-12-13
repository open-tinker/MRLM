"""PPO algorithm configuration."""

from dataclasses import dataclass


@dataclass
class PPOConfig:
    """
    Configuration for PPO training.

    Attributes:
        # Training
        num_epochs: Number of training epochs
        num_ppo_epochs: Number of PPO epochs per training iteration
        batch_size: Batch size for training
        mini_batch_size: Size of mini-batches for PPO updates
        learning_rate: Learning rate
        max_grad_norm: Maximum gradient norm for clipping

        # PPO specific
        clip_range: PPO clipping parameter (ε)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        value_loss_coef: Coefficient for value loss
        entropy_coef: Coefficient for entropy bonus
        value_clip_range: Optional value function clipping

        # Rollout collection
        num_rollouts_per_iteration: Number of rollouts to collect per iteration
        max_episode_length: Maximum episode length

        # Evaluation
        eval_every: Evaluate every N iterations
        eval_episodes: Number of episodes for evaluation
        save_every: Save checkpoint every N iterations

        # Logging
        log_every: Log metrics every N steps
    """

    # Training
    num_epochs: int = 100
    num_ppo_epochs: int = 4
    batch_size: int = 64
    mini_batch_size: int = 16
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5

    # PPO specific
    clip_range: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    value_clip_range: float = None

    # Rollout
    num_rollouts_per_iteration: int = 128
    max_episode_length: int = 512

    # Evaluation
    eval_every: int = 5
    eval_episodes: int = 10
    save_every: int = 10

    # Logging
    log_every: int = 1
