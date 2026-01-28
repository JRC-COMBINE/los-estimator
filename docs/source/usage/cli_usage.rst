Command-Line Interface
======================

The LoS Estimator can be run directly from the command line using the module entry point or by calling the provided analysis script.

Basic Usage
-----------

Run with default configuration:

.. code-block:: bash

    python -m los_estimator

Run with a custom configuration file:

.. code-block:: bash

    python -m los_estimator --config_file path/to/config.toml

Run with base config and selective overrides:

.. code-block:: bash

    python -m los_estimator --config_file los_estimator/default_config.toml --overwrite_config_file los_estimator/overwrite_config.toml

Arguments
---------

``--config_file PATH``
    Path to a complete configuration file (TOML format). If not provided, the default configuration at :file:`los_estimator/default_config.toml` is used.

``--overwrite_config_file PATH``
    Path to a partial configuration file that overrides specific settings in the base configuration. Useful for testing different model parameters without creating a full new config file.

Configuration Files
-------------------

Configuration is managed via TOML files. The package includes:

- **default_config.toml**: Default settings for model fitting, data paths, and visualization.
- **overwrite_config.toml**: Sample overrides for common adjustments.

.. include:: ../usage/_key_configs.rst

Example: Quick Test Run
-----------------------

For rapid prototyping, enable debug flags in your config or override:

.. code-block:: toml

    # overwrite_config.toml
    [debug_config]
    less_windows = true
    less_distros = true

Then run:

.. code-block:: bash

    python -m los_estimator --overwrite_config_file overwrite_config.toml

This limits fitting to ~3 windows and 2 distributions, completing in seconds.

Alternative: Using run_analysis.py
-----------------------------------

A convenience script is provided:

.. code-block:: bash

    python run_analysis.py

This script applies the same analysis pipeline using configurations from the package. Edit :file:`run_analysis.py` directly to adjust settings programmatically if preferred.

Output Structure
----------------

See :doc:`output_format` for complete documentation of all output artifacts.
