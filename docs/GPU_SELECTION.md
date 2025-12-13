# GPU Selection Guide

This guide explains all the ways to select and manage GPUs in MRLM.

## Quick Start

### 1. Environment Variable (Simplest)

```bash
# Use GPUs 0 and 2
export CUDA_VISIBLE_DEVICES=0,2
python train.py

# Use only GPU 1
export CUDA_VISIBLE_DEVICES=1
python train.py

# Use CPU only
export CUDA_VISIBLE_DEVICES=""
python train.py
```

### 2. YAML Configuration

```yaml
# config.yaml
distributed:
  num_gpus: 2          # Use 2 GPUs
  gpu_ids: [0, 2]      # Specifically GPUs 0 and 2
  strategy: "ddp"      # Or "fsdp"
```

```bash
mrlm train --config config.yaml
```

### 3. CLI Arguments

```bash
# Use 2 GPUs
mrlm train --config config.yaml --num-gpus 2

# Use specific GPUs
mrlm train --config config.yaml --gpu-ids 0 2 3

# Force CPU
mrlm train --config config.yaml --device cpu
```

### 4. Python API

```python
from mrlm.distributed import setup_gpu_environment

# Auto-select 2 GPUs
device = setup_gpu_environment(num_gpus=2)

# Use specific GPUs
device = setup_gpu_environment(gpu_ids=[0, 2])

# Use all available GPUs
device = setup_gpu_environment()
```

---

## Detailed Usage

### Method 1: Environment Variables

**Best for**: Quick testing, system-wide GPU restrictions

Set before running any Python code:

```bash
# Linux/Mac
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Windows (cmd)
set CUDA_VISIBLE_DEVICES=0,1,2,3

# Windows (PowerShell)
$env:CUDA_VISIBLE_DEVICES="0,1,2,3"
```

**Pros**:
- Simple and universal
- Works with all Python scripts
- No code changes needed

**Cons**:
- Must set before each run
- Affects all CUDA programs in session

### Method 2: YAML Configuration

**Best for**: Reproducible experiments, sharing configs

```yaml
# config.yaml
experiment_name: my_experiment

distributed:
  enabled: true
  
  # GPU selection
  num_gpus: 4                    # Use 4 GPUs
  gpu_ids: [0, 1, 2, 3]         # Specific GPU IDs
  
  # Parallelism strategy
  strategy: "ddp"                # ddp or fsdp
  
  # For FSDP (large models)
  fsdp_sharding_strategy: "full_shard"
  fsdp_cpu_offload: false

model:
  model_name_or_path: "Qwen/Qwen2.5-7B"
  device_map: "auto"             # Or "cuda", "cpu"
  torch_dtype: "bfloat16"
```

**Usage**:
```bash
mrlm train --config config.yaml
```

**Pros**:
- Configuration versioned with code
- Easy to share and reproduce
- Declarative and readable

**Cons**:
- Requires config file
- Less flexible for quick tests

### Method 3: Python API

**Best for**: Programmatic control, custom scripts

#### Basic Selection

```python
from mrlm.distributed import (
    setup_gpu_environment,
    get_available_gpus,
    select_gpus,
    print_gpu_info,
)

# Check available GPUs
available = get_available_gpus()
print(f"Available GPUs: {available}")  # [0, 1, 2, 3, 4, 5, 6, 7]

# Print GPU information
print_gpu_info()
# Output:
# Available GPUs: 8
# GPU 0: NVIDIA A100-SXM4-40GB (40536 MB total, 1234 MB used (3.0%), Compute 8.0)
# GPU 1: NVIDIA A100-SXM4-40GB (40536 MB total, 0 MB used (0.0%), Compute 8.0)
# ...

# Select specific GPUs
device = setup_gpu_environment(gpu_ids=[0, 2, 4, 6])

# Select first N GPUs
device = setup_gpu_environment(num_gpus=4)
```

#### Advanced Selection

```python
from mrlm.distributed import (
    select_gpus,
    auto_select_gpus_for_model,
    get_gpu_memory_usage,
)

# Select GPUs with at least 32GB memory
gpus = select_gpus(min_memory_mb=32000)
print(f"High-memory GPUs: {gpus}")

# Auto-select for model size
# For a 70B parameter model (~140GB)
gpus = auto_select_gpus_for_model(model_size_gb=140)
print(f"GPUs for 70B model: {gpus}")

# Check memory usage
for gpu_id in [0, 1, 2, 3]:
    allocated, total, util = get_gpu_memory_usage(gpu_id)
    print(f"GPU {gpu_id}: {allocated}/{total} MB ({util:.1f}%)")
```

#### Manual Control

```python
from mrlm.distributed import set_visible_gpus, get_optimal_device
import torch

# Set visible GPUs manually
set_visible_gpus([0, 2, 4, 6])

# After this, PyTorch only sees these GPUs
# torch.cuda.device_count() will return 4

# Get optimal device
device = get_optimal_device(gpu_ids=[0, 2, 4, 6])
print(device)  # cuda:0 (mapped to physical GPU 0)

# Move model to device
model = model.to(device)
```

---

## Common Scenarios

### Scenario 1: Single GPU Training

**Option A: Environment variable**
```bash
export CUDA_VISIBLE_DEVICES=0
python train.py
```

**Option B: Config**
```yaml
distributed:
  enabled: false
  num_gpus: 1
  gpu_ids: [0]
```

**Option C: Python**
```python
device = setup_gpu_environment(gpu_ids=[0])
model.to(device)
```

### Scenario 2: Multi-GPU DDP (2-8 GPUs)

**Using 4 GPUs with DDP:**

```yaml
# config.yaml
distributed:
  enabled: true
  num_gpus: 4
  gpu_ids: [0, 1, 2, 3]
  strategy: "ddp"
```

```bash
# Launch with torchrun
torchrun --nproc_per_node=4 \
    examples/train_distributed_ppo.py \
    --config config.yaml
```

**Or with environment variable:**
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 train.py
```

### Scenario 3: Large Model FSDP (70B+)

**Using all 8 GPUs with FSDP:**

```yaml
# config.yaml
distributed:
  enabled: true
  num_gpus: 8
  strategy: "fsdp"
  fsdp_sharding_strategy: "full_shard"
  fsdp_cpu_offload: false  # Set true if running out of memory

model:
  model_name_or_path: "meta-llama/Llama-2-70b-hf"
  torch_dtype: "bfloat16"
```

```bash
torchrun --nproc_per_node=8 train.py --config config.yaml
```

### Scenario 4: Multi-Node Training

**2 nodes, 8 GPUs each (16 total):**

**Node 0 (master):**
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    train.py --config config.yaml
```

**Node 1:**
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    train.py --config config.yaml
```

### Scenario 5: Subset of GPUs on Multi-GPU Server

**Use only GPUs 2, 3, 5, 7 on an 8-GPU server:**

```bash
export CUDA_VISIBLE_DEVICES=2,3,5,7

torchrun --nproc_per_node=4 train.py
```

Or in Python:
```python
from mrlm.distributed import setup_gpu_environment

# These GPUs will be remapped to 0,1,2,3 internally
device = setup_gpu_environment(gpu_ids=[2, 3, 5, 7])
```

### Scenario 6: Testing on CPU

**Force CPU execution:**

```bash
export CUDA_VISIBLE_DEVICES=""
python train.py
```

Or:
```python
import torch
device = torch.device("cpu")
model.to(device)
```

---

## Best Practices

### 1. Check GPU Availability First

```python
from mrlm.distributed import get_available_gpus, print_gpu_info

# Always check before training
available = get_available_gpus()
if not available:
    print("Warning: No GPUs available, using CPU")
else:
    print(f"Found {len(available)} GPUs")
    print_gpu_info()
```

### 2. Monitor GPU Memory

```python
from mrlm.distributed import get_gpu_memory_usage

# Before and after loading model
for gpu_id in gpu_ids:
    alloc, total, util = get_gpu_memory_usage(gpu_id)
    if util > 90:
        print(f"Warning: GPU {gpu_id} at {util:.1f}% capacity")
```

### 3. Use CUDA_VISIBLE_DEVICES for Isolation

```bash
# Terminal 1: Use GPUs 0-3
export CUDA_VISIBLE_DEVICES=0,1,2,3
python experiment1.py

# Terminal 2: Use GPUs 4-7
export CUDA_VISIBLE_DEVICES=4,5,6,7
python experiment2.py
```

### 4. Auto-Select for Model Size

```python
from mrlm.distributed import auto_select_gpus_for_model

# For 7B model (~14GB with gradients/optimizer)
gpus = auto_select_gpus_for_model(model_size_gb=14)

# For 70B model (~140GB)
gpus = auto_select_gpus_for_model(model_size_gb=140)
# Will use FSDP across all GPUs if needed
```

### 5. Consistent Configuration

```yaml
# Always specify in config for reproducibility
distributed:
  num_gpus: 4
  gpu_ids: [0, 1, 2, 3]  # Explicit is better than implicit
  strategy: "ddp"
```

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Solutions:**
1. Use fewer GPUs: `num_gpus: 2` instead of 4
2. Use FSDP instead of DDP
3. Enable CPU offload: `fsdp_cpu_offload: true`
4. Reduce batch size
5. Use mixed precision: `torch_dtype: "bfloat16"`

### Issue: "RuntimeError: CUDA error: invalid device ordinal"

**Cause:** Requesting GPU that doesn't exist

**Solution:**
```python
# Check available GPUs first
from mrlm.distributed import get_available_gpus
print(get_available_gpus())  # [0, 1, 2, 3]

# Don't request GPU 4 if you only have 4 GPUs (0-3)
```

### Issue: GPUs not being used

**Check:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
```

**Common causes:**
- CUDA not installed
- CUDA_VISIBLE_DEVICES set to empty string
- Model not moved to GPU: `model.to(device)`

### Issue: "Address already in use" in distributed training

**Solution:**
```bash
# Use different port
torchrun --master_port=29501 train.py
```

Or in config:
```yaml
distributed:
  master_port: 29501
```

---

## Examples

### Example 1: Interactive GPU Selection

```python
from mrlm.distributed import get_available_gpus, setup_gpu_environment

# Show available GPUs
gpus = get_available_gpus()
print(f"Available GPUs: {gpus}")

# Let user choose
gpu_input = input("Enter GPU IDs (comma-separated): ")
selected = [int(x.strip()) for x in gpu_input.split(",")]

# Setup environment
device = setup_gpu_environment(gpu_ids=selected)
print(f"Using device: {device}")
```

### Example 2: Automatic Selection Based on Load

```python
from mrlm.distributed import get_gpu_info, get_gpu_memory_usage

# Find least loaded GPU
gpu_info = get_gpu_info()
loads = []

for info in gpu_info:
    _, _, util = get_gpu_memory_usage(info['id'])
    loads.append((info['id'], util))

# Sort by utilization
loads.sort(key=lambda x: x[1])

# Use least loaded GPU
best_gpu = loads[0][0]
print(f"Using least loaded GPU: {best_gpu} ({loads[0][1]:.1f}% used)")

device = setup_gpu_environment(gpu_ids=[best_gpu])
```

### Example 3: Conditional GPU Selection

```python
from mrlm.distributed import select_gpus, setup_gpu_environment

# Training config
model_size_gb = 7  # 7B model
batch_size = 32

# Estimate memory needed
memory_needed_gb = model_size_gb * 2  # Model + gradients + optimizer

# Select GPUs
if memory_needed_gb <= 40:
    # Fits on one A100
    device = setup_gpu_environment(num_gpus=1)
    strategy = "single"
elif memory_needed_gb <= 160:
    # Use DDP on 4 GPUs
    device = setup_gpu_environment(num_gpus=4)
    strategy = "ddp"
else:
    # Need FSDP
    device = setup_gpu_environment(num_gpus=8)
    strategy = "fsdp"

print(f"Using {strategy} strategy on {device}")
```

---

## API Reference

See full API documentation at [docs/api/distributed.rst](../api/distributed.rst)

**Key Functions:**
- `get_available_gpus()` - List available GPU IDs
- `select_gpus(gpu_ids, num_gpus, min_memory_mb)` - Select GPUs by criteria
- `set_visible_gpus(gpu_ids)` - Set CUDA_VISIBLE_DEVICES
- `setup_gpu_environment(gpu_ids, num_gpus)` - All-in-one setup
- `get_gpu_info()` - Detailed GPU information
- `print_gpu_info(gpu_ids)` - Print GPU stats
- `auto_select_gpus_for_model(model_size_gb)` - Auto-select for model

---

## Summary

**For quick tests**: Use environment variables
```bash
export CUDA_VISIBLE_DEVICES=0,1
```

**For experiments**: Use YAML config
```yaml
distributed:
  num_gpus: 4
  gpu_ids: [0, 1, 2, 3]
```

**For custom scripts**: Use Python API
```python
device = setup_gpu_environment(num_gpus=4)
```

**For production**: Combine config + environment variables for flexibility
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mrlm train --config production_config.yaml
```
