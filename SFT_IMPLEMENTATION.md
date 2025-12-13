
# SFT (Supervised Fine-Tuning) Implementation ✅

**Added:** 2025-12-12
**Status:** Complete with Distributed Training Support

---

## Overview

Added **SFT (Supervised Fine-Tuning)** support for world model training and behavioral cloning. The LLM can now learn to predict future states from environment trajectories, functioning as a world model or learning from demonstrations.

## Key Features

✅ **Three Training Modes:**
1. **Behavioral Cloning**: Train model to predict actions given observations (state → action)
2. **World Model**: Train model to predict next observations (state + action → next state)
3. **Combined**: Train both objectives simultaneously with configurable weights

✅ **Environment Integration:**
- Seamlessly collects trajectories from any environment
- Works with all existing environments (code, math, debate, tools)
- Automatic trajectory collection during training
- Optional filtering by reward threshold

✅ **Distributed Training Support:**
- Full FSDP/DDP support (same as PPO, GRPO, DPO)
- Multi-GPU training
- Multi-node training
- Mixed precision training

✅ **Flexible Usage:**
- Pre-training before RL fine-tuning
- Regularization alongside RL training
- Standalone supervised learning from demonstrations
- World model learning for planning

---

## Implementation Details

### Files Created

**Core Components:**
- `src/mrlm/algorithms/sft/__init__.py` - Module exports
- `src/mrlm/algorithms/sft/dataset.py` - Trajectory dataset (~300 lines)
- `src/mrlm/algorithms/sft/loss.py` - SFT loss functions (~200 lines)
- `src/mrlm/algorithms/sft/trainer.py` - SFT trainer with distributed support (~350 lines)

**Configuration:**
- Added `SFTConfig` to `src/mrlm/config/training_config.py`
- Added `sft` field to `ExperimentConfig`

**Examples:**
- `examples/train_sft_world_model.py` - Complete training example
- `configs/sft_world_model.yaml` - YAML configuration

**Total:** ~900 lines of new code

---

## Components

### 1. Trajectory Dataset (`dataset.py`)

**`Trajectory` Dataclass:**
```python
@dataclass
class Trajectory:
    observations: List[Observation]
    actions: List[Message]
    rewards: List[Reward]
    metadata: Dict[str, Any]
```

**`TrajectoryDataset` Class:**
- Store and manage trajectories
- Save/load from JSON
- Split into train/val
- Filter by reward
- Collect from environments automatically
- Get transitions for training

**Key Methods:**
- `add_trajectory()` - Add single trajectory
- `collect_from_environment()` - Collect using policy
- `filter_by_reward()` - Keep only high-reward trajectories
- `get_transitions()` - Get (s, a, s', r) tuples
- `save()/load()` - Persistence

---

### 2. Loss Functions (`loss.py`)

**Behavioral Cloning Loss:**
```python
compute_behavioral_cloning_loss(
    model, tokenizer, observations, actions, device
) -> (loss, info)
```
- Train model to predict actions given observations
- Only computes loss on action tokens (prompt tokens masked)
- Returns perplexity and loss metrics

**World Model Loss:**
```python
compute_next_state_loss(
    model, tokenizer, observations, actions, next_observations, device
) -> (loss, info)
```
- Train model to predict next observation
- Given current state + action, predict what happens next
- Useful for planning and simulation

**Combined Loss:**
```python
compute_combined_sft_loss(
    model, tokenizer, observations, actions, next_observations, device,
    bc_weight=0.5, world_model_weight=0.5
) -> (loss, info)
```
- Weighted combination of both losses
- Configurable weights for each objective

---

### 3. SFT Trainer (`trainer.py`)

**`SFTTrainer` Class:**
Extends `BaseTrainer` with SFT-specific functionality.

**Key Features:**
- **Trajectory Collection**: Automatically collects from environments
- **Flexible Training**: Supports all three modes (BC, WM, combined)
- **Reward Filtering**: Optional filtering of low-reward trajectories
- **Distributed**: Full FSDP/DDP support
- **Evaluation**: Standard evaluation on environments

**Training Loop:**
```python
for iteration in range(num_iterations):
    # Collect trajectories every N iterations
    if iteration % collect_every == 0:
        new_trajectories = collect_rollouts()
        trajectory_dataset.add(new_trajectories)

    # Optional: Filter by reward
    if filter_low_reward:
        trajectory_dataset = filter_by_reward(min_threshold)

    # Train on trajectories
    train_metrics = train_epoch()

    # Evaluate
    eval_metrics = evaluate()
```

---

## Configuration

### SFTConfig Dataclass

```python
@dataclass
class SFTConfig:
    # Training mode
    mode: str = "combined"  # behavioral_cloning, world_model, combined

    # Loss weights (for combined mode)
    bc_weight: float = 0.5
    world_model_weight: float = 0.5

    # Trajectory filtering
    filter_low_reward: bool = False
    min_reward_threshold: float = 0.0

    # Collection
    collect_every: int = 1
    max_trajectories: int = 1000
```

### YAML Example

```yaml
sft:
  mode: combined
  bc_weight: 0.5
  world_model_weight: 0.5
  filter_low_reward: true
  min_reward_threshold: 0.5
  collect_every: 5
```

---

## Usage Examples

### 1. Behavioral Cloning

Learn to imitate expert actions:

```python
from mrlm.algorithms.sft import SFTTrainer, TrajectoryDataset
from mrlm.config.training_config import SFTConfig

config = ExperimentConfig(
    sft=SFTConfig(
        mode="behavioral_cloning",
        filter_low_reward=True,  # Only learn from successful trajectories
        min_reward_threshold=0.8,
    )
)

trainer = SFTTrainer(policy_env, eval_envs, config)
trainer.train(num_iterations=100)
```

### 2. World Model Learning

Learn to predict future states:

```python
config = ExperimentConfig(
    sft=SFTConfig(
        mode="world_model",
        collect_every=10,  # Collect new data every 10 iterations
    )
)

trainer = SFTTrainer(policy_env, eval_envs, config)
trainer.train(num_iterations=100)
```

### 3. Combined Training

Train both objectives:

```python
config = ExperimentConfig(
    sft=SFTConfig(
        mode="combined",
        bc_weight=0.6,  # 60% behavioral cloning
        world_model_weight=0.4,  # 40% world model
    )
)
```

### 4. Pre-training for RL

Use SFT to pre-train before RL fine-tuning:

```python
# Step 1: SFT pre-training
sft_trainer = SFTTrainer(policy_env, eval_envs, sft_config)
sft_trainer.train(num_iterations=50)

# Step 2: RL fine-tuning
ppo_trainer = PPOTrainer(policy_env, eval_envs, ppo_config)
ppo_trainer.train(num_iterations=100)
```

### 5. Distributed Training

Launch with torchrun for multi-GPU:

```bash
torchrun --nproc_per_node=4 examples/train_sft_world_model.py
```

Or use FSDP for large models:

```python
from mrlm.distributed import setup_for_distributed_training

model = setup_for_distributed_training(model, strategy="fsdp")
trainer = SFTTrainer(policy_env, eval_envs, config)
trainer.train()
```

---

## Integration with Existing Algorithms

SFT complements the existing RL algorithms:

| Algorithm | Type | Best For | SFT Use Case |
|-----------|------|----------|--------------|
| **PPO** | On-policy RL | General RL training | Pre-training or regularization |
| **GRPO** | Group-based RL | Variance reduction | Pre-training with demonstrations |
| **DPO** | Offline preferences | Preference learning | N/A (different paradigm) |
| **SFT** | Supervised | Behavioral cloning, world model | Pre-training, imitation, planning |

**Combined Workflow:**
1. **SFT**: Pre-train on expert demonstrations or environment interactions
2. **PPO/GRPO**: Fine-tune with reinforcement learning
3. **DPO**: Final alignment with human preferences

---

## Benefits

✅ **Improved Sample Efficiency:**
- Learn from successful trajectories
- Bootstrap RL training with supervised learning
- Warm-start policy for faster convergence

✅ **World Model Capabilities:**
- Predict consequences of actions
- Enable planning and simulation
- Improve decision-making

✅ **Flexible Training:**
- Works with any environment
- Configurable objectives
- Optional reward filtering

✅ **Production Ready:**
- Distributed training support
- Checkpoint saving/loading
- Trajectory persistence
- Comprehensive logging

---

## Algorithm Summary

**MRLM now supports 4 training algorithms:**

1. **PPO** - Proximal Policy Optimization (on-policy RL)
2. **GRPO** - Group Relative Policy Optimization (variance-reduced RL)
3. **DPO** - Direct Preference Optimization (offline preference learning)
4. **SFT** - Supervised Fine-Tuning (behavioral cloning & world model) ✨ **NEW**

All algorithms support:
- ✅ Distributed training (FSDP/DDP)
- ✅ YAML configuration
- ✅ Multi-environment support
- ✅ Comprehensive logging and checkpointing

---

## Next Steps

The SFT implementation is complete and ready to use. Potential extensions:

- **Hindsight Experience Replay**: Re-label trajectories with achieved goals
- **Trajectory Augmentation**: Data augmentation for trajectories
- **Multi-task Learning**: Train on trajectories from multiple environments
- **Online SFT**: Continuously update dataset during RL training

---

## Files Changed

**New Files:**
- `src/mrlm/algorithms/sft/__init__.py`
- `src/mrlm/algorithms/sft/dataset.py`
- `src/mrlm/algorithms/sft/loss.py`
- `src/mrlm/algorithms/sft/trainer.py`
- `examples/train_sft_world_model.py`
- `configs/sft_world_model.yaml`

**Modified Files:**
- `src/mrlm/config/training_config.py` (added SFTConfig)
- `src/mrlm/algorithms/__init__.py` (added SFT exports)

**Total Impact:** ~1,000 lines of new code
