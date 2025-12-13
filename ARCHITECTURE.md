
# MRLM Architecture Guide

This document provides a comprehensive overview of MRLM's architecture, design principles, and implementation details.

---

## Table of Contents

1. [Overview](#overview)
2. [Core Principles](#core-principles)
3. [System Architecture](#system-architecture)
4. [Component Details](#component-details)
5. [Data Flow](#data-flow)
6. [Distributed Training](#distributed-training)
7. [Extension Points](#extension-points)

---

## Overview

MRLM (Multi-Agent Reinforcement Learning for LLMs) is built on a modular, extensible architecture designed for production-scale LLM training. The system consists of several key layers:

```
┌─────────────────────────────────────────────────────────────┐
│                      Application Layer                       │
│  (CLI, Examples, User Code)                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                    Algorithm Layer                           │
│  (PPO, GRPO, DPO, SFT Trainers)                            │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                  Environment Layer                           │
│  (Code, Math, Debate, Tools, Custom)                        │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                      Core Layer                              │
│  (Types, BaseEnvironment, LLMEnvironment)                   │
└─────────────────────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                Infrastructure Layer                          │
│  (gRPC, Distributed, Config, Data)                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Principles

### 1. **Unified Interface**

All environments (LLMs and simulated) implement the same interface:

```python
class BaseEnvironment:
    def reset(self) -> Observation
    def step(self, action: Message) -> Tuple[Observation, Reward]
    def close(self)
```

**Benefits:**
- Algorithm-agnostic trainers
- Easy environment composition
- Simplified testing and debugging

### 2. **Server-Client Decomposition**

Environments operate in two modes:

- **SERVER Mode**: Environment is trainable (gradients flow)
- **CLIENT Mode**: Environment is frozen (inference only)

```python
# Training LLM (SERVER mode)
policy_env = LLMEnvironment(model, tokenizer, mode=EnvironmentMode.SERVER)

# Evaluation environment (CLIENT mode)
eval_env = CodeExecutionEnvironment(mode=EnvironmentMode.CLIENT)
```

**Benefits:**
- Clear separation of training and inference
- Distributed environment hosting
- Resource optimization

### 3. **gRPC Communication**

All remote environment access uses gRPC:

```
┌───────────────┐          gRPC          ┌────────────────┐
│ Local Trainer │ ←─────────────────────→ │ Remote Env     │
│               │                         │ (any machine)  │
└───────────────┘                         └────────────────┘
```

**Benefits:**
- Language-agnostic (environments can be written in any language)
- High performance with streaming support
- Production-ready reliability
- Built-in load balancing

### 4. **Type Safety**

Comprehensive type hints throughout:

```python
def step(self, action: Message) -> Tuple[Observation, Reward]:
    ...
```

**Benefits:**
- Catch errors at development time
- Better IDE support
- Self-documenting code
- Easier maintenance

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Application                         │
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │ CLI Tool    │    │ Python API  │    │ Config YAML │        │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘        │
└─────────┼──────────────────┼──────────────────┼────────────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
┌────────────────────────────▼───────────────────────────────────┐
│                         Trainer Layer                           │
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────┐  ┌────────┐ │
│  │ PPOTrainer  │  │ GRPOTrainer  │  │DPOTrainer│  │SFTTrainer│
│  └─────┬───────┘  └──────┬───────┘  └────┬─────┘  └────┬───┘ │
└────────┼─────────────────┼───────────────┼─────────────┼──────┘
         │                 │               │             │
         └─────────────────┼───────────────┼─────────────┘
                           │               │
┌──────────────────────────▼───────────────▼───────────────────┐
│                    Environment Layer                          │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ LLMEnvironment  │  CodeEnv  │  │  MathEnv    │      │
│  │ (Policy)     │  │ (Eval)    │  │  (Eval)     │      │
│  └──────┬───────┘  └──────┬────┘  └──────┬──────┘      │
└─────────┼─────────────────┼───────────────┼─────────────────┘
          │                 │               │
┌─────────▼─────────────────▼───────────────▼─────────────────┐
│                       Core Layer                              │
│                                                               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │   Types    │  │    Base    │  │ Generation │            │
│  │  (Message, │  │Environment │  │  Utils     │            │
│  │  Reward)   │  │            │  │            │            │
│  └────────────┘  └────────────┘  └────────────┘            │
└───────────────────────────────────────────────────────────────┘
          │                 │               │
┌─────────▼─────────────────▼───────────────▼─────────────────┐
│                  Infrastructure Layer                         │
│                                                               │
│  ┌────────┐  ┌────────────┐  ┌──────────┐  ┌────────────┐  │
│  │ gRPC   │  │ Distributed│  │  Config  │  │   Data     │  │
│  │ Server │  │ (FSDP/DDP) │  │  Parser  │  │  Buffer    │  │
│  └────────┘  └────────────┘  └──────────┘  └────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Core Layer

#### Types (`src/mrlm/core/types.py`)

Fundamental data structures:

```python
@dataclass
class Message:
    """A message in a conversation."""
    role: Union[MessageRole, str]  # system, user, assistant
    content: str
    metadata: Dict[str, Any]

@dataclass
class Observation:
    """Environment observation."""
    messages: List[Message]  # Conversation history
    state: Optional[Dict[str, Any]]  # Additional state info
    done: bool  # Episode termination flag
    info: Dict[str, Any]  # Extra information

@dataclass
class Reward:
    """Reward signal."""
    value: float  # Total reward
    components: Dict[str, float]  # Reward decomposition
    info: Dict[str, Any]  # Metadata

@dataclass
class RolloutBatch:
    """Batch of rollout data for training."""
    observations: List[Observation]
    actions: List[Message]
    rewards: List[Reward]
    log_probs: Optional[torch.Tensor]
    values: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    returns: Optional[torch.Tensor]
```

#### BaseEnvironment (`src/mrlm/core/base.py`)

Abstract interface for all environments:

```python
class BaseEnvironment(ABC):
    """Base class for all environments."""

    def __init__(self, mode: EnvironmentMode):
        self.mode = mode  # SERVER or CLIENT

    @abstractmethod
    def reset(self) -> Observation:
        """Reset environment to initial state."""
        pass

    @abstractmethod
    def step(self, action: Message) -> Tuple[Observation, Reward]:
        """Execute action and return next state and reward."""
        pass

    @abstractmethod
    def close(self):
        """Clean up resources."""
        pass
```

#### LLMEnvironment (`src/mrlm/core/llm_environment.py`)

Wraps HuggingFace models as environments:

```python
class LLMEnvironment(BaseEnvironment):
    """LLM wrapped as an environment."""

    def __init__(self, model, tokenizer, mode: EnvironmentMode):
        super().__init__(mode)
        self.model = model
        self.tokenizer = tokenizer

        if mode == EnvironmentMode.SERVER:
            self.model.train()  # Enable gradients
        else:
            self.model.eval()  # Freeze model

    def reset(self) -> Observation:
        """Start new conversation."""
        return Observation(messages=[], done=False)

    def step(self, action: Message) -> Tuple[Observation, Reward]:
        """Generate response to user message."""
        # Generate with/without gradients based on mode
        response = self._generate(action)
        return Observation([action, response]), Reward(0.0)
```

### 2. Algorithm Layer

All algorithms inherit from `BaseTrainer`:

```python
class BaseTrainer(ABC):
    """Base class for all RL trainers."""

    def __init__(self, policy_env, eval_envs, config, device):
        self.policy_env = policy_env  # LLM being trained
        self.eval_envs = eval_envs  # Evaluation environments
        self.config = config
        self.device = device

    @abstractmethod
    def collect_rollouts(self) -> RolloutBuffer:
        """Collect trajectories from environments."""
        pass

    @abstractmethod
    def train_epoch(self, rollouts) -> Dict[str, float]:
        """Train for one epoch on rollouts."""
        pass

    def train(self, num_iterations: int):
        """Main training loop."""
        for iteration in range(num_iterations):
            rollouts = self.collect_rollouts()
            metrics = self.train_epoch(rollouts)
            # Log, evaluate, save...
```

#### PPOTrainer

```python
class PPOTrainer(BaseTrainer):
    def collect_rollouts(self):
        """Collect trajectories using current policy."""
        buffer = RolloutBuffer()
        for env in self.eval_envs:
            obs = env.reset()
            while not done:
                action, log_prob, value = self._generate_action_with_value(obs)
                next_obs, reward = env.step(action)
                buffer.add(obs, action, reward, log_prob, value)
        buffer.compute_gae()  # Compute advantages
        return buffer

    def train_epoch(self, rollouts):
        """PPO update on collected data."""
        for epoch in range(self.config.num_ppo_epochs):
            for batch in rollouts.get_batches():
                # Recompute log probs and values
                current_log_probs, current_values = self._evaluate(batch)

                # PPO losses
                policy_loss = compute_ppo_loss(
                    current_log_probs, batch.old_log_probs, batch.advantages
                )
                value_loss = compute_value_loss(current_values, batch.returns)

                # Update
                loss = policy_loss + value_loss
                loss.backward()
                self.optimizer.step()
```

### 3. Environment Layer

#### Simulated Environments

All task environments extend `SimulatedEnvironment`:

```python
class SimulatedEnvironment(BaseEnvironment):
    """Base class for simulated (non-LLM) environments."""

    def __init__(self, mode: EnvironmentMode = EnvironmentMode.CLIENT):
        super().__init__(mode)
        # Simulated environments are always frozen (CLIENT mode)
```

Example: Math Environment

```python
class MathReasoningEnvironment(SimulatedEnvironment):
    def __init__(self, problem_generator):
        super().__init__(EnvironmentMode.CLIENT)
        self.generator = problem_generator
        self.current_problem = None

    def reset(self) -> Observation:
        """Generate new math problem."""
        self.current_problem = self.generator.generate()
        message = Message(
            role=MessageRole.USER,
            content=f"Solve: {self.current_problem.question}"
        )
        return Observation(messages=[message], done=False)

    def step(self, action: Message) -> Tuple[Observation, Reward]:
        """Evaluate answer."""
        answer = self._parse_answer(action.content)
        is_correct = self._check_answer(answer, self.current_problem.answer)

        reward = Reward(
            value=1.0 if is_correct else 0.0,
            components={"correctness": 1.0 if is_correct else 0.0}
        )

        return Observation(messages=[], done=True), reward
```

### 4. Infrastructure Layer

#### gRPC Server (`src/mrlm/server/`)

Hosts environments remotely:

```protobuf
// mrlm.proto
service EnvironmentService {
    rpc Reset(ResetRequest) returns (ResetResponse);
    rpc Step(StepRequest) returns (StepResponse);
    rpc BatchStep(BatchStepRequest) returns (BatchStepResponse);
    rpc StreamStep(stream StepRequest) returns (stream StepResponse);
}
```

```python
class EnvironmentServer:
    def __init__(self, environments: Dict[str, BaseEnvironment]):
        self.environments = environments
        self.sessions = {}  # session_id -> (env_id, env)

    def Reset(self, request, context):
        """Create new session and reset environment."""
        env_id = request.environment_id
        session_id = str(uuid.uuid4())
        env = self.environments[env_id]

        obs = env.reset()
        self.sessions[session_id] = (env_id, env)

        return mrlm_pb2.ResetResponse(
            observation=self._to_proto(obs),
            session_id=session_id
        )
```

#### Distributed Training (`src/mrlm/distributed/`)

**FSDP Support:**
```python
def setup_fsdp_model(model, sharding_strategy="full_shard"):
    """Wrap model with Fully Sharded Data Parallel."""
    from torch.distributed.fsdp import FullyShardedDataParallel

    # Auto-detect transformer layers
    transformer_layer_cls = get_transformer_layer_cls(model)

    # Configure FSDP
    fsdp_config = {
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "auto_wrap_policy": transformer_auto_wrap_policy(transformer_layer_cls),
        "mixed_precision": MixedPrecision(param_dtype=torch.float16),
    }

    return FullyShardedDataParallel(model, **fsdp_config)
```

**DDP Support:**
```python
def setup_ddp_model(model, device_ids=None):
    """Wrap model with Distributed Data Parallel."""
    from torch.nn.parallel import DistributedDataParallel

    if device_ids is None:
        device_ids = [get_local_rank()]

    return DistributedDataParallel(
        model,
        device_ids=device_ids,
        find_unused_parameters=False,
    )
```

---

## Data Flow

### Training Loop Data Flow

```
1. Rollout Collection
   ┌──────────────┐
   │ Policy (LLM) │
   └──────┬───────┘
          │ Generate actions
          ├──────────────────────────────┐
          ▼                              ▼
   ┌──────────────┐              ┌──────────────┐
   │  Env 1       │              │  Env 2       │
   │  (Code)      │              │  (Math)      │
   └──────┬───────┘              └──────┬───────┘
          │ Observations + Rewards       │
          └──────────────┬───────────────┘
                         ▼
                  ┌──────────────┐
                  │RolloutBuffer │
                  │(obs, actions,│
                  │ rewards, ...)│
                  └──────┬───────┘

2. Advantage Computation
          ┌──────────────┐
          │RolloutBuffer │
          └──────┬───────┘
                 │ Compute GAE
                 ▼
          ┌──────────────┐
          │ Advantages + │
          │   Returns    │
          └──────┬───────┘

3. Training
          ┌──────────────┐
          │  Batches     │
          └──────┬───────┘
                 │ For each batch
                 ▼
          ┌──────────────┐
          │ Recompute    │
          │ log probs    │
          └──────┬───────┘
                 │
                 ▼
          ┌──────────────┐
          │ Compute Loss │
          │ (PPO/GRPO)   │
          └──────┬───────┘
                 │
                 ▼
          ┌──────────────┐
          │   Backward   │
          │   Update     │
          └──────────────┘
```

### Distributed Training Data Flow

```
Node 0                          Node 1
┌─────────────────┐            ┌─────────────────┐
│ Process 0       │            │ Process 2       │
│ ┌─────────────┐ │            │ ┌─────────────┐ │
│ │Model Shard 0│ │            │ │Model Shard 2│ │
│ └──────┬──────┘ │            │ └──────┬──────┘ │
│        │        │            │        │        │
│ ┌──────▼──────┐ │            │ ┌──────▼──────┐ │
│ │ Gradients   │ │            │ │ Gradients   │ │
│ └──────┬──────┘ │            │ └──────┬──────┘ │
└────────┼────────┘            └────────┼────────┘
         │                              │
         └──────────┬───────────────────┘
                    │ All-Reduce
                    ▼
         ┌──────────────────┐
         │ Synced Gradients │
         └──────────────────┘
```

---

## Extension Points

### Adding a New Algorithm

1. **Create algorithm directory:**
   ```
   src/mrlm/algorithms/my_algo/
   ├── __init__.py
   ├── loss.py
   ├── trainer.py
   └── config.py (optional)
   ```

2. **Implement trainer:**
   ```python
   class MyAlgoTrainer(BaseTrainer):
       def collect_rollouts(self):
           # Custom rollout collection
           pass

       def train_epoch(self, rollouts):
           # Custom training logic
           pass
   ```

3. **Add configuration:**
   ```python
   @dataclass
   class MyAlgoConfig:
       param1: float = 0.1
       param2: int = 10
   ```

4. **Export in `__init__.py`:**
   ```python
   from mrlm.algorithms.my_algo import MyAlgoTrainer
   __all__ = ["MyAlgoTrainer"]
   ```

### Adding a New Environment

1. **Create environment class:**
   ```python
   class MyEnvironment(SimulatedEnvironment):
       def reset(self) -> Observation:
           # Initialize task
           pass

       def step(self, action: Message) -> Tuple[Observation, Reward]:
           # Process action, return reward
           pass
   ```

2. **Add to environment registry:**
   ```python
   # src/mrlm/environments/__init__.py
   from mrlm.environments.my_env import MyEnvironment
   __all__ = [..., "MyEnvironment"]
   ```

3. **Update CLI utils:**
   ```python
   # src/mrlm/cli/utils.py
   def create_environment_by_name(name):
       if name == "my_env":
           return MyEnvironment(...)
   ```

### Adding Custom Rewards

Reward shaping through `Reward.components`:

```python
reward = Reward(
    value=total_reward,
    components={
        "correctness": 0.8,
        "efficiency": 0.1,
        "style": 0.1,
    },
    info={"details": "..."}
)
```

---

## Performance Considerations

### Memory Optimization

1. **Use FSDP for large models:**
   - Shards parameters across GPUs
   - Reduces memory per GPU by ~N× (N = num GPUs)

2. **Gradient accumulation:**
   - Simulate larger batches without OOM
   - Trade compute for memory

3. **Mixed precision:**
   - fp16/bf16 reduces memory by 2×
   - Maintains model quality

### Compute Optimization

1. **Batch operations:**
   - Collect multiple rollouts in parallel
   - Process batches efficiently

2. **Distributed rollouts:**
   - Use gRPC to distribute environments
   - Scale collection across machines

3. **Caching:**
   - Cache tokenization results
   - Reuse KV cache in generation

---

## Security Considerations

### Code Execution

The `CodeExecutionEnvironment` uses sandboxing:

```python
# Remove dangerous builtins
safe_globals = {
    "__builtins__": {
        k: v for k, v in __builtins__.items()
        if k not in ["open", "exec", "eval", "compile", "input"]
    }
}

# Execute with timeout
with time_limit(5):
    exec(code, safe_globals)
```

### Tool Use

The `ToolUseEnvironment` restricts file access:

```python
# Restrict to workspace directory
def file_operation(operation, path, content):
    full_path = (workspace_dir / path).resolve()
    if not str(full_path).startswith(str(workspace_dir)):
        return "Error: Access denied"
    # Proceed with operation
```

---

## Conclusion

MRLM's architecture is designed for:
- **Modularity**: Easy to extend with new algorithms and environments
- **Scalability**: Distributed training from single GPU to multi-node
- **Reliability**: Production-ready with comprehensive error handling
- **Flexibility**: Unified interface supports diverse use cases

For more details, see:
- [Main README](README.md)
- [Examples](examples/README.md)
- [API Documentation](docs/api/)
