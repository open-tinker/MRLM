MRLM: Multi-Agent Reinforcement Learning for LLMs
==================================================

**MRLM** is a comprehensive, production-ready library for training Large Language Models using multi-agent reinforcement learning. Built with a clean server-client architecture, full distributed training support, and state-of-the-art RL algorithms.

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.9+

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
   :alt: License

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: black

Features
--------

* **🤖 4 RL Algorithms**: PPO, GRPO, DPO, and SFT for diverse training scenarios
* **🌍 4 Task Environments**: Code execution, math reasoning, multi-agent debate, and tool use
* **⚡ Distributed Training**: Full FSDP and DDP support for multi-GPU/multi-node training
* **🏗️ Server-Client Architecture**: gRPC-based distributed environment hosting
* **📋 YAML Configuration**: Declarative configuration system for reproducible experiments
* **🔧 Professional CLI**: Complete command-line interface for training, serving, and evaluation
* **🎯 Production Ready**: Type hints, comprehensive docs, and extensive examples

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   # From PyPI (coming soon)
   pip install mrlm

   # From source
   git clone https://github.com/youjiaxuan/MRLM.git
   cd MRLM
   pip install -e .

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

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

CLI Quick Start
~~~~~~~~~~~~~~~

.. code-block:: bash

   # Train from config
   mrlm train --config configs/code_generation_ppo.yaml

   # Start environment server
   mrlm serve --environments code,math,debate --port 50051

   # Evaluate trained model
   mrlm eval --model outputs/ppo/final --environment math --num-episodes 20

   # Show system info
   mrlm info

User Guide
----------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   tutorials

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   algorithms
   environments
   distributed
   configuration
   cli

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/algorithms
   api/environments
   api/distributed
   api/config
   api/cli

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   architecture
   examples
   contributing
   changelog

Algorithms
----------

MRLM implements 4 state-of-the-art algorithms for LLM training:

* **PPO (Proximal Policy Optimization)**: General-purpose on-policy RL with clipped surrogate objective
* **GRPO (Group Relative Policy Optimization)**: Variance reduction through group-normalized rewards
* **DPO (Direct Preference Optimization)**: Offline preference learning without reward models
* **SFT (Supervised Fine-Tuning)**: Behavioral cloning and world model training

Environments
------------

Built-in environments for diverse tasks:

* **Code Execution**: Train LLMs to write and debug code with test-based rewards
* **Math Reasoning**: Solve mathematical problems with automatic answer verification
* **Multi-Agent Debate**: Structured debates with argument quality evaluation
* **Tool Use**: Learn to use external tools (calculator, web search, Python REPL, filesystem)

Distributed Training
--------------------

Full support for large-scale training:

* **FSDP (Fully Sharded Data Parallel)**: For very large models (10B+ parameters)
* **DDP (Distributed Data Parallel)**: For smaller models with full replication
* **Mixed Precision**: fp16/bf16 for 2x speedup
* **Multi-Node**: Scale across multiple machines

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
