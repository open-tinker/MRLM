Quick Start
===========

This guide will get you started with MRLM in 5 minutes.

30-Second Example
-----------------

Here's a minimal example to train a model with PPO on math reasoning:

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

Save this as ``train.py`` and run:

.. code-block:: bash

   python train.py

Training from Configuration
---------------------------

For production use, train from YAML configs:

1. Create a configuration file ``my_config.yaml``:

   .. code-block:: yaml

      experiment_name: my_first_experiment

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
        - env_type: math
          mode: client
          max_turns: 5

2. Train using the CLI:

   .. code-block:: bash

      mrlm train --config my_config.yaml

Using Different Algorithms
---------------------------

PPO (Proximal Policy Optimization)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

General-purpose on-policy RL:

.. code-block:: python

   from mrlm.algorithms.ppo import PPOTrainer

   trainer = PPOTrainer(policy_env, eval_envs, config)
   trainer.train(num_iterations=100)

GRPO (Group Relative Policy Optimization)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Better for high-variance tasks:

.. code-block:: python

   from mrlm.algorithms.grpo import GRPOTrainer
   from mrlm.config import GRPOConfig

   config.grpo = GRPOConfig(group_size=4, clip_range=0.2)
   trainer = GRPOTrainer(policy_env, eval_envs, config)
   trainer.train(num_iterations=100)

DPO (Direct Preference Optimization)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Offline training on preference pairs:

.. code-block:: python

   from mrlm.algorithms.dpo import DPOTrainer, PreferenceDataset

   # Load preference dataset
   dataset = PreferenceDataset.load("preferences.json")

   trainer = DPOTrainer(policy_env, dataset, config)
   trainer.train(num_iterations=100)

SFT (Supervised Fine-Tuning)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pre-training with behavioral cloning and world model:

.. code-block:: python

   from mrlm.algorithms.sft import SFTTrainer
   from mrlm.config import SFTConfig

   config.sft = SFTConfig(
       mode="combined",
       bc_weight=0.6,
       world_model_weight=0.4
   )

   trainer = SFTTrainer(policy_env, eval_envs, config)
   trainer.train(num_iterations=50)

Using Different Environments
-----------------------------

Code Execution
~~~~~~~~~~~~~~

Train on code generation tasks:

.. code-block:: python

   from mrlm.environments.code import CodeExecutionEnvironment, CodeProblemGenerator

   generator = CodeProblemGenerator()
   env = CodeExecutionEnvironment(generator, mode=EnvironmentMode.CLIENT)

Math Reasoning
~~~~~~~~~~~~~~

Train on mathematical problems:

.. code-block:: python

   from mrlm.environments.math import MathReasoningEnvironment, MathProblemGenerator

   generator = MathProblemGenerator(difficulty_range=(1, 3))
   env = MathReasoningEnvironment(generator, mode=EnvironmentMode.CLIENT)

Multi-Agent Debate
~~~~~~~~~~~~~~~~~~

Train on structured debates:

.. code-block:: python

   from mrlm.environments.debate import DebateEnvironment, RuleBasedJudge

   judge = RuleBasedJudge()
   env = DebateEnvironment(judge=judge, mode=EnvironmentMode.CLIENT)

Tool Use
~~~~~~~~

Train on tool use tasks:

.. code-block:: python

   from mrlm.environments.tools import ToolUseEnvironment
   from mrlm.environments.tools.builtin_tools import create_default_tool_registry

   registry = create_default_tool_registry()
   env = ToolUseEnvironment(registry, mode=EnvironmentMode.CLIENT)

CLI Quick Reference
-------------------

Train
~~~~~

.. code-block:: bash

   mrlm train --config CONFIG [--output DIR] [--resume CHECKPOINT]

Serve Environments
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   mrlm serve --environments code,math,debate --port 50051

Evaluate Model
~~~~~~~~~~~~~~

.. code-block:: bash

   mrlm eval --model MODEL --environment ENV --num-episodes N

Collect Trajectories
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   mrlm collect --model MODEL --environment ENV -n N -o OUTPUT.json

System Information
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   mrlm info

Distributed Training
--------------------

Single Node, Multiple GPUs
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Using DDP
   torchrun --nproc_per_node=4 train.py --strategy ddp

   # Using FSDP (for large models)
   torchrun --nproc_per_node=4 train.py --strategy fsdp

Multi-Node Training
~~~~~~~~~~~~~~~~~~~

Node 0 (master):

.. code-block:: bash

   torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
       --master_addr=192.168.1.1 --master_port=12355 \
       train.py --strategy fsdp

Node 1:

.. code-block:: bash

   torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
       --master_addr=192.168.1.1 --master_port=12355 \
       train.py --strategy fsdp

Next Steps
----------

* Explore :doc:`examples` for comprehensive examples
* Read the :doc:`architecture` guide to understand the system design
* Check out the :doc:`api/core` for detailed API reference
* Visit :doc:`contributing` to contribute to MRLM
