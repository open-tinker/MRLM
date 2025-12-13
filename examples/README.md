# MRLM Examples

This directory contains example scripts demonstrating how to use MRLM for training LLMs with reinforcement learning.

## 📚 Table of Contents

- [Quick Start Examples](#quick-start-examples)
- [Algorithm Examples](#algorithm-examples)
- [Environment Examples](#environment-examples)
- [Advanced Examples](#advanced-examples)
- [Configuration-Based Training](#configuration-based-training)
- [Distributed Training](#distributed-training)

---

## Quick Start Examples

### `quickstart/simple_ppo.py`
**Simplest possible PPO training example**

```bash
python examples/quickstart/simple_ppo.py
```

A minimal example showing PPO training on a simple math environment. Great starting point to understand the basics.

---

## Algorithm Examples

### PPO (Proximal Policy Optimization)

#### `train_code_ppo.py`
Train on code generation tasks with PPO.

```bash
python examples/train_code_ppo.py
```

**Features:**
- Code execution environment
- Test case validation
- Syntax and functionality rewards

#### `train_math_ppo.py`
Train on mathematical reasoning with PPO.

```bash
python examples/train_math_ppo.py
```

**Features:**
- Arithmetic, algebra, and word problems
- Answer parsing and validation
- Difficulty progression

### GRPO (Group Relative Policy Optimization)

#### `train_grpo_math.py`
Train with GRPO for variance reduction.

```bash
python examples/train_grpo_math.py
```

**Features:**
- Group-normalized rewards
- Multiple responses per prompt
- Reduced variance in updates

**Key Differences from PPO:**
- Generates multiple responses per prompt (group_size=4)
- Normalizes rewards within groups
- Better for high-variance environments

### DPO (Direct Preference Optimization)

#### `train_dpo_preferences.py`
Offline training on preference pairs.

```bash
python examples/train_dpo_preferences.py
```

**Features:**
- Preference pair dataset
- No online rollouts needed
- Direct preference optimization

**Use Cases:**
- Alignment from human feedback
- Learning from demonstrations
- Preference-based fine-tuning

### SFT (Supervised Fine-Tuning)

#### `train_sft_world_model.py`
World model training and behavioral cloning.

```bash
python examples/train_sft_world_model.py
```

**Features:**
- Behavioral cloning (state → action)
- World model (state + action → next state)
- Combined training
- Trajectory filtering

**Training Modes:**
- `behavioral_cloning`: Learn to imitate actions
- `world_model`: Learn to predict future states
- `combined`: Train both objectives

---

## Environment Examples

### Code Execution

```bash
python examples/train_code_ppo.py
```

**Tasks:**
- Function implementation
- Algorithm coding
- Test case passing

### Math Reasoning

```bash
python examples/train_math_ppo.py
```

**Problem Types:**
- Arithmetic (addition, subtraction, multiplication, division)
- Algebra (equations, inequalities)
- Word problems

### Tool Use

#### `train_tool_use_ppo.py`
Train agent to use external tools.

```bash
python examples/train_tool_use_ppo.py
```

**Available Tools:**
- Calculator (math operations)
- Web Search (simulated knowledge base)
- Python REPL (code execution)
- File System (read/write)

**Sample Tasks:**
- "What is 15% of 280?" → Uses calculator
- "How tall is the Eiffel Tower?" → Uses web search
- "First 10 Fibonacci numbers?" → Uses Python REPL

---

## Advanced Examples

### Multi-Environment Training

#### `train_multi_environment.py`
Train on multiple task types simultaneously.

```bash
python examples/train_multi_environment.py
```

**Environments:**
- Code execution (2 instances)
- Math reasoning (2 instances)
- Tool use (2 instances)

**Benefits:**
- Learn to generalize across tasks
- Single model for multiple capabilities
- Improved sample efficiency

### Hybrid Training (SFT → PPO)

#### `train_hybrid_sft_ppo.py`
Two-stage training pipeline.

```bash
python examples/train_hybrid_sft_ppo.py
```

**Pipeline:**
1. **Stage 1 (SFT)**: Pre-train on trajectories (30 epochs)
   - Behavioral cloning
   - World model learning
   - Filter high-reward trajectories

2. **Stage 2 (PPO)**: Fine-tune with RL (50 epochs)
   - Optimize for rewards
   - Refine policy
   - Lower learning rate

**Benefits:**
- Better initialization from SFT
- Faster convergence in PPO
- Higher final performance

---

## Configuration-Based Training

### `train_from_config.py`
Train from YAML configuration files.

```bash
python examples/train_from_config.py --config configs/code_generation_ppo.yaml
```

**Available Configs:**
- `configs/code_generation_ppo.yaml` - Code generation with PPO
- `configs/math_reasoning_ppo.yaml` - Math reasoning with PPO
- `configs/math_grpo.yaml` - Math with GRPO
- `configs/preference_dpo.yaml` - DPO training
- `configs/sft_world_model.yaml` - SFT world model
- `configs/quick_start.yaml` - Minimal quick start

**Advantages:**
- Reproducible experiments
- Easy hyperparameter tuning
- Share configurations
- Version control

---

## Distributed Training

### Single Node, Multiple GPUs

#### `train_distributed_ppo.py`
Multi-GPU training with DDP or FSDP.

**DDP (Distributed Data Parallel):**
```bash
torchrun --nproc_per_node=4 examples/train_distributed_ppo.py --strategy ddp
```

**FSDP (Fully Sharded Data Parallel) - For large models:**
```bash
torchrun --nproc_per_node=4 examples/train_distributed_ppo.py --strategy fsdp
```

### Multi-Node Training

**Node 0 (master):**
```bash
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr=<node0_ip> --master_port=12355 \
    examples/train_distributed_ppo.py --strategy fsdp
```

**Node 1:**
```bash
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
    --master_addr=<node0_ip> --master_port=12355 \
    examples/train_distributed_ppo.py --strategy fsdp
```

**Distributed Features:**
- ✅ FSDP for very large models (> 10B parameters)
- ✅ DDP for smaller models
- ✅ Mixed precision training
- ✅ Gradient accumulation
- ✅ Multi-node support

---

## CLI Tool

MRLM provides a command-line interface for common operations:

### Train
```bash
mrlm train --config configs/code_generation_ppo.yaml
```

### Serve Environments
```bash
mrlm serve --environments code,math,debate --port 50051
```

### Evaluate
```bash
mrlm eval --model outputs/ppo/final --environment code --num-episodes 10
```

### Collect Trajectories
```bash
mrlm collect --model Qwen/Qwen2.5-1.5B --environment math \
    --num-episodes 100 --output trajectories.json
```

### System Info
```bash
mrlm info
```

---

## Example Workflows

### 1. Quick Experimentation
```bash
# Start with a simple example
python examples/quickstart/simple_ppo.py

# Try different algorithms
python examples/train_math_ppo.py
python examples/train_grpo_math.py
```

### 2. Serious Training
```bash
# Use configuration files
mrlm train --config configs/code_generation_ppo.yaml

# Distributed training for scale
torchrun --nproc_per_node=8 examples/train_distributed_ppo.py --strategy fsdp
```

### 3. Hybrid Pipeline
```bash
# SFT pre-training + PPO fine-tuning
python examples/train_hybrid_sft_ppo.py

# Or step by step:
mrlm train --config configs/sft_world_model.yaml
mrlm train --config configs/math_reasoning_ppo.yaml --resume outputs/sft/final
```

### 4. Multi-Task Training
```bash
# Train on multiple environments
python examples/train_multi_environment.py

# Results in a generalist model
```

---

## Tips and Best Practices

### Hyperparameter Tuning

**Learning Rate:**
- PPO: 5e-6 to 3e-5
- GRPO: 1e-5 to 5e-5
- DPO: 1e-6 to 1e-5
- SFT: 5e-6 to 1e-5

**Batch Size:**
- Small models: 8-16
- Medium models: 16-32
- Large models: 32-64 (with gradient accumulation)

**Clipping Range (PPO/GRPO):**
- Start with 0.2
- Increase to 0.3 for more aggressive updates
- Decrease to 0.1 for more stable training

### Environment Selection

**Code Execution:**
- Best for: Code generation, debugging
- Difficulty: Medium
- Sample efficiency: Low (sparse rewards)

**Math Reasoning:**
- Best for: Numerical reasoning, word problems
- Difficulty: Medium-High
- Sample efficiency: Medium

**Tool Use:**
- Best for: Planning, multi-step reasoning
- Difficulty: High
- Sample efficiency: Medium

**Multi-Environment:**
- Best for: Generalization, robustness
- Difficulty: High
- Sample efficiency: High (learns from diverse data)

### Algorithm Selection

| Scenario | Recommended Algorithm |
|----------|----------------------|
| General RL training | **PPO** |
| High variance tasks | **GRPO** |
| Preference data available | **DPO** |
| Limited data / cold start | **SFT** → PPO |
| Multi-task learning | **PPO** or **GRPO** |

---

## Troubleshooting

### Out of Memory
- Use FSDP instead of DDP
- Reduce batch size
- Enable gradient accumulation
- Use smaller model

### Slow Training
- Enable mixed precision (fp16/bf16)
- Use distributed training
- Increase batch size
- Reduce evaluation frequency

### Poor Performance
- Try hybrid SFT → PPO pipeline
- Adjust learning rate
- Increase training epochs
- Filter low-reward trajectories (SFT)

### Model Not Learning
- Check reward function
- Verify environment is solvable
- Increase exploration (entropy_coef)
- Lower learning rate

---

## Next Steps

1. **Start Simple**: Run `quickstart/simple_ppo.py`
2. **Explore Algorithms**: Try PPO, GRPO, DPO, SFT
3. **Try Environments**: Code, math, tools, debate
4. **Scale Up**: Use distributed training
5. **Optimize**: Use YAML configs and hybrid pipelines

For more information, see:
- [Main README](../README.md)
- [Installation Guide](../INSTALL.md)
- [Contributing Guide](../CONTRIBUTING.md)
- [API Documentation](https://mrlm.readthedocs.io)
