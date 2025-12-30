# MRLM: Multi-Agent Reinforcement Learning for LLMs

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**MRLM** is a comprehensive, production-ready library for training Large Language Models using multi-agent reinforcement learning. Built with a clean server-client architecture, full distributed training support, and state-of-the-art RL algorithms.

## ✨ Key Features

- **🤖 4 RL Algorithms**: PPO, GRPO, DPO, and SFT for diverse training scenarios
- **🌍 4 Task Environments**: Code execution, math reasoning, multi-agent debate, and tool use
- **⚡ Distributed Training**: Full FSDP and DDP support for multi-GPU/multi-node training
- **🏗️ Server-Client Architecture**: gRPC-based distributed environment hosting
- **📋 YAML Configuration**: Declarative configuration system for reproducible experiments
- **🔧 Professional CLI**: Complete command-line interface for training, serving, and evaluation
- **🎯 Production Ready**: Type hints, comprehensive docs, and extensive examples

## 🚀 Quick Start

### Installation

```bash
# From PyPI (coming soon)
pip install mrlm

# From source
git clone https://github.com/open-tinker/MRLM.git
cd MRLM
pip install -e .
```

### 30-Second Example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mrlm.core import LLMEnvironment, EnvironmentMode
from mrlm.environments.math import MathReasoningEnvironment, MathProblemGenerator
from mrlm.algorithms.ppo import PPOTrainer
from mrlm.config import ExperimentConfig, TrainingConfig, PPOConfig

# Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

# Create environments
policy_env = LLMEnvironment(model, tokenizer, mode=EnvironmentMode.SERVER)
eval_envs = [MathReasoningEnvironment(MathProblemGenerator()) for _ in range(4)]

# Configure and train
config = ExperimentConfig(
    training=TrainingConfig(algorithm="ppo", num_epochs=50),
    ppo=PPOConfig(clip_range=0.2, gamma=0.99),
)

trainer = PPOTrainer(policy_env, eval_envs, config)
trainer.train(num_iterations=50)
```

### CLI Quick Start

```bash
# Train from config
mrlm train --config configs/code_generation_ppo.yaml

# Start environment server
mrlm serve --environments code,math,debate --port 50051

# Evaluate trained model
mrlm eval --model outputs/ppo/final --environment math --num-episodes 20

# Collect trajectories for SFT
mrlm collect --model Qwen/Qwen2.5-1.5B --environment code -n 100 -o data/trajectories.json

# Show system info
mrlm info
```

## 🤖 Algorithms

MRLM implements 4 state-of-the-art algorithms for LLM training:

| Algorithm | Type | Best For | Key Features |
|-----------|------|----------|--------------|
| **PPO** | On-policy RL | General RL training | Clipped surrogate, GAE, stable updates |
| **GRPO** | Group-based RL | Variance reduction | Group-normalized rewards, efficient sampling |
| **DPO** | Offline preferences | Human alignment | No reward model, direct preference optimization |
| **SFT** | Supervised | Pre-training, world model | Behavioral cloning, next-state prediction |

### Algorithm Comparison

```python
# PPO - Most versatile, general-purpose
ppo_trainer = PPOTrainer(policy_env, eval_envs, config)

# GRPO - Better for high-variance tasks
grpo_trainer = GRPOTrainer(policy_env, eval_envs, config)

# DPO - Train on preference pairs
dpo_trainer = DPOTrainer(policy_env, preference_dataset, config)

# SFT - Pre-train on trajectories
sft_trainer = SFTTrainer(policy_env, eval_envs, config)
```

All algorithms support:
- ✅ Distributed training (FSDP/DDP)
- ✅ Mixed precision (fp16/bf16)
- ✅ Gradient accumulation
- ✅ YAML configuration
- ✅ Checkpointing and resumption

## 🌍 Environments

### Built-in Environments

#### 1. Code Execution
Train LLMs to write and debug code with test-based rewards.

```python
from mrlm.environments.code import CodeExecutionEnvironment, CodeProblemGenerator

generator = CodeProblemGenerator()
env = CodeExecutionEnvironment(generator)
```

**Features:**
- Python code execution with sandboxing
- Test case validation
- Syntax and functionality scoring
- Support for HumanEval-style problems

#### 2. Math Reasoning
Solve mathematical problems with automatic answer verification.

```python
from mrlm.environments.math import MathReasoningEnvironment, MathProblemGenerator

generator = MathProblemGenerator(difficulty_range=(1, 3))
env = MathReasoningEnvironment(generator)
```

**Problem Types:**
- Arithmetic (addition, subtraction, multiplication, division)
- Algebra (equations, inequalities)
- Word problems
- Multi-step reasoning

#### 3. Multi-Agent Debate
Train agents to engage in structured debates.

```python
from mrlm.environments.debate import DebateEnvironment, RuleBasedJudge

judge = RuleBasedJudge()
env = DebateEnvironment(judge=judge)
```

**Features:**
- PRO/CON position assignment
- Argument quality evaluation
- Evidence and reasoning scoring
- Consistency metrics

#### 4. Tool Use
Learn to use external tools for complex tasks.

```python
from mrlm.environments.tools import ToolUseEnvironment
from mrlm.environments.tools.builtin_tools import create_default_tool_registry

registry = create_default_tool_registry()  # Calculator, web search, Python REPL, filesystem
env = ToolUseEnvironment(registry)
```

**Built-in Tools:**
- Calculator (math operations)
- Web Search (knowledge retrieval)
- Python REPL (code execution)
- File System (read/write)

### Custom Environments

Create custom environments by extending `SimulatedEnvironment`:

```python
from mrlm.core import SimulatedEnvironment
from mrlm.core.types import Message, Observation, Reward

class MyEnvironment(SimulatedEnvironment):
    def reset(self) -> Observation:
        # Initialize task
        return Observation(messages=[Message(...)])
    
    def step(self, action: Message) -> tuple[Observation, Reward]:
        # Process action, compute reward
        return next_observation, reward
```

## 🏗️ Architecture

### Server-Client Model

MRLM uses a clean server-client architecture for distributed training:

```
┌─────────────────┐      gRPC      ┌─────────────────┐
│  Policy Model   │ ←────────────→ │  Environment 1  │
│   (Training)    │                │    (Client)     │
│   SERVER Mode   │      gRPC      ├─────────────────┤
│                 │ ←────────────→ │  Environment 2  │
└─────────────────┘                │    (Client)     │
                                   └─────────────────┘
```

**Benefits:**
- Scale environments independently
- Distribute compute across machines
- Hot-swap environments without restarting training
- Clear separation of concerns

### Distributed Training

Full support for large-scale training:

```bash
# Single node, 4 GPUs with FSDP
torchrun --nproc_per_node=4 train.py --strategy fsdp

# Multi-node, 2 nodes × 4 GPUs
# Node 0:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr=<node0_ip> --master_port=12355 train.py

# Node 1:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
    --master_addr=<node0_ip> --master_port=12355 train.py
```

**Strategies:**
- **DDP** (Distributed Data Parallel): Best for models < 10B parameters
- **FSDP** (Fully Sharded Data Parallel): For very large models (10B+)
- **Mixed Precision**: fp16/bf16 for 2x speedup
- **Gradient Accumulation**: Simulate larger batches

## 📋 Configuration System

Use YAML configs for reproducible experiments:

```yaml
# configs/my_experiment.yaml
experiment_name: code_generation_ppo

training:
  algorithm: ppo
  num_epochs: 100
  batch_size: 16
  learning_rate: 5.0e-6

ppo:
  clip_range: 0.2
  gamma: 0.99
  gae_lambda: 0.95

model:
  model_name_or_path: "Qwen/Qwen2.5-1.5B-Instruct"
  torch_dtype: "float16"

eval_envs:
  - env_type: code
    mode: client
    max_turns: 3

distributed:
  enabled: true
  strategy: "fsdp"
```

Train with:
```bash
mrlm train --config configs/my_experiment.yaml
```

## 📚 Examples

The `examples/` directory contains 11+ comprehensive examples:

### Basic Examples
- `quickstart/simple_ppo.py` - Minimal PPO example
- `train_code_ppo.py` - Code generation with PPO
- `train_math_ppo.py` - Math reasoning with PPO
- `train_grpo_math.py` - GRPO training
- `train_dpo_preferences.py` - DPO from preferences
- `train_sft_world_model.py` - SFT for world model
- `train_tool_use_ppo.py` - Tool use training

### Advanced Examples
- `train_distributed_ppo.py` - Multi-GPU distributed training
- `train_multi_environment.py` - Train on code, math, and tools simultaneously
- `train_hybrid_sft_ppo.py` - Hybrid pipeline (SFT pre-train → PPO fine-tune)
- `train_from_config.py` - Config-based training

See [`examples/README.md`](examples/README.md) for detailed guide.

## 🔧 CLI Reference

### Train
```bash
mrlm train --config CONFIG [--output DIR] [--resume CHECKPOINT]
```

Train a model from YAML configuration. Supports all algorithms (PPO, GRPO, DPO, SFT).

### Serve
```bash
mrlm serve --environments ENV1,ENV2,... [--port PORT] [--host HOST]
```

Start a gRPC server hosting multiple environments for distributed training.

### Eval
```bash
mrlm eval --model MODEL --environment ENV [--num-episodes N] [--output FILE]
```

Evaluate a trained model on an environment and save results.

### Collect
```bash
mrlm collect --model MODEL --environment ENV --num-episodes N --output FILE [--filter-reward THRESHOLD]
```

Collect trajectory data for SFT training.

### Info
```bash
mrlm info
```

Display system information, available algorithms, and environments.

## 🎓 Tutorials & Guides

### Tutorial 1: Quick Start with PPO
```python
# See examples/quickstart/simple_ppo.py
# A minimal, self-contained example showing PPO training
```

### Tutorial 2: Multi-Environment Training
```python
# See examples/train_multi_environment.py
# Train a generalist model on multiple task types
```

### Tutorial 3: Hybrid Training Pipeline
```python
# See examples/train_hybrid_sft_ppo.py
# Best-practice two-stage training: SFT → PPO
```

### Tutorial 4: Distributed Training
```bash
# See examples/train_distributed_ppo.py
# Scale to multiple GPUs with FSDP or DDP
```

## 📖 Documentation

Comprehensive documentation available:

- **[Architecture Guide](ARCHITECTURE.md)** - System design and components
- **[Examples README](examples/README.md)** - Complete guide to all examples
- **[Installation Guide](INSTALL.md)** - Detailed installation instructions
- **[Contributing Guide](CONTRIBUTING.md)** - Development guidelines
- **[API Reference](docs/api/)** - Auto-generated API documentation

## 🛠️ Development

### Setup Development Environment

```bash
git clone https://github.com/open-tinker/MRLM.git
cd MRLM
pip install -e ".[dev]"
pre-commit install
```

### Running Tests

```bash
pytest                          # Run all tests
pytest --cov=mrlm              # With coverage
pytest -m "not slow"           # Skip slow tests
```

### Code Quality

```bash
black src/ tests/              # Format code
ruff check src/ tests/         # Lint
mypy src/                      # Type checking
```

## 🎯 Use Cases

### 1. Code Generation
Train models to write, debug, and explain code with test-based rewards.

### 2. Math Reasoning
Improve mathematical problem-solving with structured reasoning rewards.

### 3. Tool Use & Planning
Teach models to use external tools and plan multi-step solutions.

### 4. Dialogue & Debate
Train conversational agents through multi-agent debate and evaluation.

### 5. Pre-training with SFT
Bootstrap RL training with supervised fine-tuning on demonstrations.

### 6. Preference Alignment
Align models with human preferences using DPO (no reward model needed).

## 🌟 Highlights

### Why MRLM?

**Compared to other RL libraries for LLMs:**

| Feature | MRLM | TRL | VeRL | Others |
|---------|------|-----|------|--------|
| **Algorithms** | PPO, GRPO, DPO, SFT | PPO, DPO | PPO, GRPO | Varies |
| **Distributed** | FSDP, DDP, Multi-node | Limited | FSDP, DDP | Varies |
| **Server-Client** | ✅ gRPC | ❌ | ✅ | ❌ |
| **Environments** | 4 built-in + custom | Limited | Custom only | Varies |
| **CLI Tool** | ✅ Full-featured | ❌ | ❌ | ❌ |
| **YAML Config** | ✅ | ❌ | ❌ | ❌ |
| **Production Ready** | ✅ | Partial | ✅ | Varies |

**MRLM is unique in providing:**
- Complete environment suite out-of-the-box
- Professional CLI for production workflows
- Full server-client architecture for distributed environments
- 4 algorithms in one unified framework
- Production-ready with extensive docs and examples

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas for contribution:**
- New environments (text summarization, translation, etc.)
- Additional RL algorithms (SAC, TD3, etc.)
- Performance optimizations
- Documentation improvements
- Bug reports and feature requests

## 📊 Benchmarks

Performance on standard benchmarks (coming soon):

| Task | Algorithm | Score | Training Time |
|------|-----------|-------|---------------|
| HumanEval | PPO | TBD | TBD |
| GSM8K | GRPO | TBD | TBD |
| MT-Bench | DPO | TBD | TBD |

## 📄 Citation

If you use MRLM in your research, please cite:

```bibtex
@software{mrlm2024,
  title={MRLM: Multi-Agent Reinforcement Learning for LLMs},
  author={MRLM Contributors},
  year={2024},
  url={https://github.com/open-tinker/MRLM},
  version={0.1.0}
}
```

## 📜 License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

MRLM is inspired by and builds upon ideas from:
- [VeRL](https://github.com/volcengine/verl) - Flexible RL framework for LLMs
- [TRL](https://github.com/huggingface/trl) - Transformer Reinforcement Learning
- [OpenAI Gym](https://github.com/openai/gym) - RL environment interface
- [Ray RLlib](https://docs.ray.io/en/latest/rllib/) - Distributed RL

## 💬 Support

- **Issues**: [GitHub Issues](https://github.com/open-tinker/MRLM/issues)
- **Discussions**: [GitHub Discussions](https://github.com/open-tinker/MRLM/discussions)
- **Email**: mrlm-dev@example.com

## 🗺️ Roadmap

- [x] Core RL algorithms (PPO, GRPO, DPO, SFT)
- [x] Built-in environments (Code, Math, Debate, Tools)
- [x] Distributed training (FSDP, DDP)
- [x] CLI tool and YAML configs
- [ ] Comprehensive test suite
- [ ] PyPI release
- [ ] Benchmark results
- [ ] Web UI for monitoring
- [ ] More environments (summarization, translation, etc.)
- [ ] More algorithms (SAC, A2C, etc.)
- [ ] Integration with popular LLM frameworks

---

**Made with ❤️ by the MRLM team**

⭐ Star us on GitHub if you find MRLM useful!
