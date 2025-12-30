Installation
============

MRLM requires Python 3.9 or higher.

From Source
-----------

The recommended way to install MRLM is from source:

.. code-block:: bash

   git clone https://github.com/open-tinker/MRLM.git
   cd MRLM
   pip install -e .

This will install MRLM in editable mode along with all required dependencies.

From PyPI (Coming Soon)
-----------------------

Once released, you'll be able to install MRLM from PyPI:

.. code-block:: bash

   pip install mrlm

Development Installation
------------------------

For development, install with additional dependencies:

.. code-block:: bash

   git clone https://github.com/open-tinker/MRLM.git
   cd MRLM
   pip install -e ".[dev]"
   pre-commit install

This installs:

* Core dependencies (PyTorch, Transformers, etc.)
* Development tools (pytest, black, ruff, mypy)
* Documentation tools (Sphinx, sphinx-rtd-theme)
* Pre-commit hooks for code quality

Dependencies
------------

Core Dependencies
~~~~~~~~~~~~~~~~~

* **Python** >= 3.9
* **PyTorch** >= 2.0.0
* **Transformers** >= 4.30.0
* **grpcio** >= 1.50.0
* **protobuf** >= 4.21.0
* **pyyaml** >= 6.0
* **numpy** >= 1.24.0
* **tqdm** >= 4.65.0

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

* **sphinx** >= 5.0.0 (for building documentation)
* **sphinx-rtd-theme** >= 1.2.0 (for documentation theme)
* **pytest** >= 7.3.0 (for running tests)
* **black** >= 23.3.0 (for code formatting)
* **ruff** >= 0.0.270 (for linting)
* **mypy** >= 1.3.0 (for type checking)

GPU Support
-----------

MRLM supports CUDA-enabled GPUs for training. To use GPU acceleration:

1. Install PyTorch with CUDA support:

   .. code-block:: bash

      # For CUDA 11.8
      pip install torch --index-url https://download.pytorch.org/whl/cu118

      # For CUDA 12.1
      pip install torch --index-url https://download.pytorch.org/whl/cu121

2. Verify GPU is available:

   .. code-block:: python

      import torch
      print(f"CUDA available: {torch.cuda.is_available()}")
      print(f"GPU count: {torch.cuda.device_count()}")

Multi-GPU and Distributed Training
-----------------------------------

For distributed training across multiple GPUs or nodes:

* **Single node, multiple GPUs**: No additional setup required
* **Multi-node**: Ensure all nodes can communicate and have the same environment

Example distributed setup:

.. code-block:: bash

   # On each node
   export MASTER_ADDR=<master_node_ip>
   export MASTER_PORT=12355

   # Launch training
   torchrun --nproc_per_node=4 --nnodes=2 --node_rank=<node_rank> \
       examples/train_distributed_ppo.py --strategy fsdp

Verification
------------

Verify your installation:

.. code-block:: bash

   # Check MRLM is installed
   python -c "import mrlm; print(mrlm.__version__)"

   # Run system info command
   mrlm info

   # Run a simple test
   python examples/quickstart/simple_ppo.py

Troubleshooting
---------------

ImportError: No module named 'mrlm'
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Make sure you installed MRLM:

.. code-block:: bash

   pip install -e .

CUDA Out of Memory
~~~~~~~~~~~~~~~~~~

If you encounter OOM errors:

* Reduce batch size in your config
* Enable gradient accumulation
* Use FSDP for large models
* Use mixed precision (fp16/bf16)

gRPC Errors
~~~~~~~~~~~

If you see gRPC-related errors:

.. code-block:: bash

   pip install --upgrade grpcio grpcio-tools protobuf

Protocol Buffer Errors
~~~~~~~~~~~~~~~~~~~~~~

Recompile the protocol buffers:

.. code-block:: bash

   cd src/mrlm/grpc_
   python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. environment.proto

Next Steps
----------

After installation, proceed to the :doc:`quickstart` guide to start training your first model.
