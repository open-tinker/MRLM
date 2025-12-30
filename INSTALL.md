# Installation Guide for MRLM

This guide covers installation and setup for the MRLM library.

## Quick Install

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/open-tinker/MRLM.git
cd MRLM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode
pip install -e ".[dev]"

# Compile protocol buffers
python -m mrlm.protocols.compile_proto
```

### From PyPI (Once Published)

```bash
pip install mrlm
```

## Requirements

- Python 3.9 or higher
- PyTorch 2.0 or higher
- CUDA (optional, for GPU training)

## Installation Steps

### 1. Install Core Dependencies

```bash
pip install torch transformers grpcio grpcio-tools protobuf pyyaml numpy tqdm
```

### 2. Compile Protocol Buffers

The gRPC communication layer requires compiling protocol buffer definitions:

```bash
# From package root
python -m mrlm.protocols.compile_proto

# Or directly
python src/mrlm/protocols/compile_proto.py
```

This generates:
- `mrlm_pb2.py` - Message definitions
- `mrlm_pb2_grpc.py` - Service stubs
- `mrlm_pb2.pyi` - Type stubs

### 3. Verify Installation

```bash
# Test basic import
python -c "import mrlm; print(mrlm.__version__)"

# Run Phase 1 test
python examples/quickstart/phase1_test.py
```

## Optional Dependencies

### For Distributed Training

```bash
pip install deepspeed  # DeepSpeed for large model training
```

### For Experiment Tracking

```bash
pip install wandb tensorboard
```

### For Development

```bash
pip install "mrlm[dev]"  # Includes pytest, black, ruff, mypy, etc.
```

### All Optional Dependencies

```bash
pip install "mrlm[all]"
```

## GPU Setup

### CUDA

For NVIDIA GPU support, install PyTorch with CUDA:

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### ROCm (AMD)

For AMD GPU support:

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm5.6
```

## Troubleshooting

### Protocol Buffer Compilation Fails

If `python -m mrlm.protocols.compile_proto` fails:

1. Ensure `grpcio-tools` is installed:
   ```bash
   pip install grpcio-tools
   ```

2. Try compiling manually:
   ```bash
   cd src/mrlm/protocols
   python -m grpc_tools.protoc \
       --proto_path=. \
       --python_out=. \
       --grpc_python_out=. \
       --pyi_out=. \
       mrlm.proto
   ```

### Import Errors

If you see `ModuleNotFoundError: No module named 'mrlm'`:

1. Ensure you're in the virtual environment
2. Install in editable mode: `pip install -e .`
3. Check Python path: `python -c "import sys; print(sys.path)"`

### CUDA Out of Memory

If training fails with CUDA OOM:

1. Reduce batch size in config
2. Enable gradient checkpointing
3. Use mixed precision (fp16/bf16)
4. Consider model parallelism

## Platform-Specific Notes

### macOS

- M1/M2 Macs: PyTorch has MPS (Metal) support for GPU acceleration
- Use `device="mps"` for Apple Silicon GPU

### Windows

- Use Command Prompt or PowerShell, not Git Bash for activation
- Some features may require WSL2 for best compatibility

### Linux

- Most features work out of the box
- Ensure GCC/G++ are installed for compiling extensions

## Minimal Example

After installation, try this minimal example:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from mrlm.core import LLMEnvironment, EnvironmentMode, Message, MessageRole

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Create environment
env = LLMEnvironment(model, tokenizer, mode=EnvironmentMode.CLIENT)

# Use it
obs = env.reset()
action = Message(role=MessageRole.USER, content="Hello!")
obs, reward = env.step(action)
print(obs.messages[-1].content)
```

## Next Steps

- Read the [Quick Start Guide](examples/quickstart/README.md)
- Browse [Examples](examples/)
- Check [Documentation](https://mrlm.readthedocs.io)
- Join [Discussions](https://github.com/open-tinker/MRLM/discussions)
