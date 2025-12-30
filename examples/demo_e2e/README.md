# MRLM Complete End-to-End Demo

This directory contains a complete, self-contained demonstration of the MRLM framework. The demo includes:

- **Synthetic data generation** for realistic testing
- **Full training pipeline** with PPO algorithm
- **Model evaluation** before and after training
- **Performance metrics** and comparisons

## Quick Start

### 1. Install Dependencies

First, ensure you have MRLM installed:

```bash
cd ../../
pip install -e .
```

### 2. Generate Synthetic Data

Generate synthetic math problems for the demo:

```bash
cd examples/demo_e2e
python synthetic_data_generator.py
```

This creates a `data/` directory with synthetic datasets:
- `math_problems.json` - 20 math word problems
- `code_problems.json` - 15 coding challenges
- `debate_topics.json` - 10 debate topics
- `tool_scenarios.json` - 8 tool usage scenarios

### 3. Run the Complete Demo

Execute the end-to-end demo:

```bash
python demo_complete.py
```

## What the Demo Does

The demo showcases the complete MRLM workflow:

### Phase 1: Setup
1. **Loads synthetic data** - Math problems with known answers
2. **Configures training** - Sets up PPO with demo-friendly parameters
3. **Loads language model** - Uses GPT-2 (124M parameters) for quick testing
4. **Creates environments** - Policy environment and evaluation environments

### Phase 2: Baseline Evaluation
5. **Evaluates untrained model** - Tests performance before training
   - Runs 3 test episodes
   - Measures accuracy and average reward

### Phase 3: Training
6. **Trains with PPO** - Reinforcement learning training
   - 5 epochs (quick demo)
   - 8 rollouts per iteration
   - Evaluates every 2 iterations

### Phase 4: Final Evaluation
7. **Evaluates trained model** - Tests performance after training
   - Runs 5 test episodes
   - Compares with baseline metrics

## Expected Output

The demo will print detailed progress through all phases:

```
======================================================================
MRLM: Complete End-to-End Demo
======================================================================

[1/7] Loading Synthetic Data
✓ Loaded 20 math problems

[2/7] Configuring Training
Configuration:
  • Epochs: 5
  • Batch Size: 4
  • Learning Rate: 1e-05
  ...

[6/7] Initial Evaluation (Before Training)
Episode 1/3:
  Problem: Alice has 25 apples. She gets 12 more apples...
  Model Answer: ...
  Correct: ✗
  Reward: 0.00
...

[7/7] Training with PPO
Starting training...
...

======================================================================
Training Summary
======================================================================
Performance Comparison:
Metric               Before          After           Change
----------------------------------------------------------------------
Accuracy              20.0%            40.0%          +20.0%
Avg Reward            0.150            0.350          +0.200
```

## Demo Features

### Quick Execution
- Uses small model (GPT-2) for fast testing
- Minimal training iterations (5 epochs)
- Small batch sizes and rollout counts
- Works on CPU (no GPU required)

### Synthetic Data
- Realistic math problems with varied difficulty
- Known correct answers for accurate evaluation
- Multiple problem types (addition, multiplication, multi-step)

### Complete Pipeline
- End-to-end workflow demonstration
- Before/after training comparison
- Detailed logging and metrics
- Error handling for robustness

## Files in This Directory

```
demo_e2e/
├── README.md                      # This file
├── synthetic_data_generator.py    # Generate synthetic datasets
├── demo_complete.py               # Complete end-to-end demo
└── data/                          # Generated data (after running generator)
    ├── math_problems.json
    ├── code_problems.json
    ├── debate_topics.json
    └── tool_scenarios.json
```

## Customization

### Use Different Model

Edit `demo_complete.py` and change the model:

```python
# Change from:
model_name = "gpt2"

# To:
model_name = "Qwen/Qwen2.5-1.5B"  # Larger, more capable model
```

### Longer Training

Adjust training parameters in `demo_complete.py`:

```python
config = PPOConfig(
    num_epochs=50,              # More training iterations
    num_rollouts_per_iteration=32,  # More rollouts
    batch_size=16,              # Larger batches
    # ... other params
)
```

### More Synthetic Data

Modify `synthetic_data_generator.py`:

```python
# Generate more problems
math_problems = generate_math_problems(num_problems=100)  # Instead of 20
code_problems = generate_code_problems(num_problems=50)   # Instead of 15
```

## Troubleshooting

### Out of Memory

If you encounter OOM errors:
- Use smaller batch size: `batch_size=2`
- Reduce rollouts: `num_rollouts_per_iteration=4`
- Use smaller model: Keep `gpt2` instead of larger models

### Slow Execution

To speed up the demo:
- Reduce epochs: `num_epochs=3`
- Reduce evaluation episodes: `num_eval_episodes=2`
- Use GPU if available (automatically detected)

### Data Not Found

If you see "Synthetic data not found":
1. Make sure you ran `python synthetic_data_generator.py` first
2. Check that `data/math_problems.json` exists
3. Run the generator from the `demo_e2e/` directory

## Next Steps

After running this demo, explore:

1. **Other Algorithms**
   - GRPO: Group Relative Policy Optimization
   - DPO: Direct Preference Optimization
   - SFT: Supervised Fine-Tuning

2. **Other Environments**
   - Code execution environment
   - Debate environment
   - Tool usage environment

3. **Distributed Training**
   - Multi-GPU training with FSDP
   - Distributed Data Parallel (DDP)

4. **Real Datasets**
   - GSM8K for math reasoning
   - HumanEval for code generation
   - Real-world debate topics

5. **Production Training**
   - Use YAML configs: `mrlm train --config config.yaml`
   - Save checkpoints regularly
   - Monitor with TensorBoard
   - Longer training runs (100+ epochs)

## Architecture Overview

This demo demonstrates MRLM's layered architecture:

```
┌─────────────────────────────────────────────────────────┐
│  Demo Application (demo_complete.py)                    │
├─────────────────────────────────────────────────────────┤
│  Algorithm Layer (PPO Trainer)                          │
├─────────────────────────────────────────────────────────┤
│  Environment Layer (Math Environment, LLM Environment)  │
├─────────────────────────────────────────────────────────┤
│  Core Layer (Types, Base Classes)                       │
├─────────────────────────────────────────────────────────┤
│  Infrastructure (PyTorch, Transformers)                 │
└─────────────────────────────────────────────────────────┘
```

## Resources

- **Main Documentation**: `../../docs/`
- **More Examples**: `../`
- **Configuration Guide**: `../../configs/`
- **API Reference**: `../../docs/api/`

## Support

For issues or questions:
- Check the main README: `../../README.md`
- Review examples: `../`
- See documentation: `../../docs/`

---

**Happy Training with MRLM! 🚀**
