Configuration Module
====================

The configuration module provides YAML-based configuration for experiments.

Training Configuration
----------------------

.. automodule:: mrlm.config.training_config
   :members:
   :undoc-members:
   :show-inheritance:

ExperimentConfig
~~~~~~~~~~~~~~~~

.. autoclass:: mrlm.config.training_config.ExperimentConfig
   :members:
   :undoc-members:
   :show-inheritance:

TrainingConfig
~~~~~~~~~~~~~~

.. autoclass:: mrlm.config.training_config.TrainingConfig
   :members:
   :undoc-members:
   :show-inheritance:

ModelConfig
~~~~~~~~~~~

.. autoclass:: mrlm.config.training_config.ModelConfig
   :members:
   :undoc-members:
   :show-inheritance:

GenerationConfig
~~~~~~~~~~~~~~~~

.. autoclass:: mrlm.config.training_config.GenerationConfig
   :members:
   :undoc-members:
   :show-inheritance:

Algorithm Configurations
------------------------

PPOConfig
~~~~~~~~~

.. autoclass:: mrlm.config.training_config.PPOConfig
   :members:
   :undoc-members:
   :show-inheritance:

GRPOConfig
~~~~~~~~~~

.. autoclass:: mrlm.config.training_config.GRPOConfig
   :members:
   :undoc-members:
   :show-inheritance:

DPOConfig
~~~~~~~~~

.. autoclass:: mrlm.config.training_config.DPOConfig
   :members:
   :undoc-members:
   :show-inheritance:

SFTConfig
~~~~~~~~~

.. autoclass:: mrlm.config.training_config.SFTConfig
   :members:
   :undoc-members:
   :show-inheritance:

Environment Configuration
-------------------------

EvalEnvConfig
~~~~~~~~~~~~~

.. autoclass:: mrlm.config.training_config.EvalEnvConfig
   :members:
   :undoc-members:
   :show-inheritance:

Distributed Configuration
-------------------------

DistributedConfig
~~~~~~~~~~~~~~~~~

.. autoclass:: mrlm.config.training_config.DistributedConfig
   :members:
   :undoc-members:
   :show-inheritance:

Configuration Loader
--------------------

.. automodule:: mrlm.config.loader
   :members:
   :undoc-members:
   :show-inheritance:

load_config
~~~~~~~~~~~

.. autofunction:: mrlm.config.loader.load_config

save_config
~~~~~~~~~~~

.. autofunction:: mrlm.config.loader.save_config
