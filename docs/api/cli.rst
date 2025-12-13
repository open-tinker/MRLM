CLI Module
==========

The CLI module provides a command-line interface for MRLM operations.

Main Entry Point
----------------

.. automodule:: mrlm.cli.main
   :members:
   :undoc-members:
   :show-inheritance:

main
~~~~

.. autofunction:: mrlm.cli.main.main

create_parser
~~~~~~~~~~~~~

.. autofunction:: mrlm.cli.main.create_parser

Commands
--------

.. automodule:: mrlm.cli.commands
   :members:
   :undoc-members:
   :show-inheritance:

Train Command
~~~~~~~~~~~~~

.. autofunction:: mrlm.cli.commands.train_command

.. autofunction:: mrlm.cli.commands.add_train_parser

Serve Command
~~~~~~~~~~~~~

.. autofunction:: mrlm.cli.commands.serve_command

.. autofunction:: mrlm.cli.commands.add_serve_parser

Eval Command
~~~~~~~~~~~~

.. autofunction:: mrlm.cli.commands.eval_command

.. autofunction:: mrlm.cli.commands.add_eval_parser

Collect Command
~~~~~~~~~~~~~~~

.. autofunction:: mrlm.cli.commands.collect_command

.. autofunction:: mrlm.cli.commands.add_collect_parser

Info Command
~~~~~~~~~~~~

.. autofunction:: mrlm.cli.commands.info_command

.. autofunction:: mrlm.cli.commands.add_info_parser

CLI Utilities
-------------

.. automodule:: mrlm.cli.utils
   :members:
   :undoc-members:
   :show-inheritance:

Environment Creation
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: mrlm.cli.utils.create_environment_by_name

.. autofunction:: mrlm.cli.utils.list_available_environments

Model Loading
~~~~~~~~~~~~~

.. autofunction:: mrlm.cli.utils.load_model_and_tokenizer

Output Formatting
~~~~~~~~~~~~~~~~~

.. autofunction:: mrlm.cli.utils.print_system_info

.. autofunction:: mrlm.cli.utils.format_eval_results
