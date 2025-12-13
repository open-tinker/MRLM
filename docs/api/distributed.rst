Distributed Training Module
===========================

The distributed module provides utilities for large-scale distributed training.

FSDP (Fully Sharded Data Parallel)
-----------------------------------

.. automodule:: mrlm.distributed.fsdp
   :members:
   :undoc-members:
   :show-inheritance:

Setup Functions
~~~~~~~~~~~~~~~

.. autofunction:: mrlm.distributed.fsdp.setup_fsdp_model

.. autofunction:: mrlm.distributed.fsdp.get_fsdp_config

DDP (Distributed Data Parallel)
--------------------------------

.. automodule:: mrlm.distributed.ddp
   :members:
   :undoc-members:
   :show-inheritance:

Setup Functions
~~~~~~~~~~~~~~~

.. autofunction:: mrlm.distributed.ddp.setup_ddp

.. autofunction:: mrlm.distributed.ddp.cleanup_ddp

Distributed Utilities
---------------------

.. automodule:: mrlm.distributed.utils
   :members:
   :undoc-members:
   :show-inheritance:

Initialization
~~~~~~~~~~~~~~

.. autofunction:: mrlm.distributed.utils.init_distributed

.. autofunction:: mrlm.distributed.utils.is_distributed

Synchronization
~~~~~~~~~~~~~~~

.. autofunction:: mrlm.distributed.utils.all_reduce

.. autofunction:: mrlm.distributed.utils.all_gather

.. autofunction:: mrlm.distributed.utils.barrier

Rank Management
~~~~~~~~~~~~~~~

.. autofunction:: mrlm.distributed.utils.get_rank

.. autofunction:: mrlm.distributed.utils.get_world_size

.. autofunction:: mrlm.distributed.utils.is_main_process
