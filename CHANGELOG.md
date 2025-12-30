# Changelog

All notable changes to MRLM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-12-12

### Initial Release

#### Added

**Core Features:**
- Unified environment interface for LLMs and simulated environments
- Server-client architecture with gRPC communication
- Full type safety with comprehensive type hints
- Message-based communication system

**Algorithms:**
- PPO (Proximal Policy Optimization) with GAE
- GRPO (Group Relative Policy Optimization) with group normalization
- DPO (Direct Preference Optimization) for offline preference learning
- SFT (Supervised Fine-Tuning) with behavioral cloning and world model training

**Environments:**
- Code Execution: Python code generation with test-based rewards
- Math Reasoning: Mathematical problem solving with answer verification
- Multi-Agent Debate: Structured debates with argument evaluation
- Tool Use: External tool usage (calculator, web search, Python REPL, filesystem)

**Distributed Training:**
- FSDP (Fully Sharded Data Parallel) support for large models (10B+)
- DDP (Distributed Data Parallel) support
- Multi-node training capabilities
- Mixed precision training (fp16/bf16)
- Gradient accumulation

**Configuration System:**
- YAML-based declarative configuration
- Type-safe configuration classes
- Config save/load with validation
- Support for all algorithms and environments

**CLI Tool:**
- `mrlm train`: Train from YAML configuration
- `mrlm serve`: Start gRPC environment server
- `mrlm eval`: Evaluate trained models
- `mrlm collect`: Collect trajectories for SFT
- `mrlm info`: Display system information

**Examples:**
- 11+ comprehensive examples covering all features
- Quick start examples
- Advanced multi-environment training
- Hybrid SFT → PPO pipeline
- Distributed training examples
- Config-based training examples

**Documentation:**
- Complete README with quick start
- Architecture guide (ARCHITECTURE.md)
- Installation guide (INSTALL.md)
- Contributing guidelines (CONTRIBUTING.md)
- API documentation with Sphinx
- Examples README with detailed guides

**Testing:**
- 200+ test cases covering all components
- Unit tests for core, algorithms, environments
- Integration tests for end-to-end workflows
- 80%+ code coverage
- CI/CD ready test suite

#### Technical Details

**Dependencies:**
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- gRPC >= 1.50.0
- Python >= 3.9

**Package Structure:**
- Modular design with clear separation of concerns
- Professional CLI with argparse
- Complete type hints throughout
- PEP 8 compliant code formatting
- Comprehensive docstrings

**Performance:**
- Optimized for multi-GPU training
- Support for models up to 70B+ parameters with FSDP
- Efficient batch processing
- Memory-optimized rollout collection

#### Known Limitations

- PyPI package not yet published (use source installation)
- Some environments may require additional setup
- GPU recommended for large model training
- Documentation hosted locally (ReadTheDocs integration pending)

#### Future Plans

- [ ] PyPI release
- [ ] ReadTheDocs hosting
- [ ] Additional environments (summarization, translation)
- [ ] More RL algorithms (SAC, A2C, TD3)
- [ ] Web UI for monitoring
- [ ] Benchmark results on standard datasets
- [ ] Integration with popular LLM frameworks

---

## [Unreleased]

### Planned for 0.2.0

- PyPI package release
- Online documentation on ReadTheDocs
- Benchmark results (HumanEval, GSM8K, MT-Bench)
- Additional environments
- Performance optimizations
- Enhanced logging and monitoring

---

**Note**: This is the initial release of MRLM. Please report any issues on [GitHub Issues](https://github.com/open-tinker/MRLM/issues).
