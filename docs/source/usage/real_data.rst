Estimation Run
======================
This example demonstrates how to use the LoS Estimator with real-world data.
We will use a sample dataset containing ICU admissions and occupancy data.

A prepared script is provided in :file:`run_analysis.py` that performs the estimation using the provided data.


Data Preparation
----------------

Using Preprocessed Data
^^^^^^^^^^^^^^^^^^^^^^^

The package includes preprocessed ICU data from Germany ready for analysis. The :file:`default_config.toml` is already configured to use these files—no additional setup is required.

Updating with Fresh Data (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use the latest data from the Robert Koch Institute (RKI):

1. **Download the data:**

   The RKI maintains COVID-19 ICU data at the `Intensivregister repository <https://github.com/robert-koch-institut/Intensivkapazitaeten_und_COVID-19-Intensivbettenbelegung_in_Deutschland/blob/main/Intensivregister_Bundeslaender_Kapazitaeten.csv>`_.

2. **Place the CSV file:**

   Save the downloaded file to :file:`los_estimator/data/preprocessing/inputs`

3. **Run the preprocessing script:**

   .. code-block:: bash

       python los_estimator/data/preprocessing/__init__.py

   This generates the required csv file.

4. **Verify configuration:**

   Ensure :file:`default_config.toml` points to the newly generated files.


Running the Analysis
--------------------

Using the Analysis Script
^^^^^^^^^^^^^^^^^^^^^^^^^

Run the provided convenience script:

.. code-block:: bash

    python run_analysis.py

**Note:** By default, ``less_windows`` is disabled in the script, allowing analysis of all windows. To perform a quick test with fewer windows, edit the script and set ``less_windows = True``.

Using the Command Line
^^^^^^^^^^^^^^^^^^^^^^

Alternatively, use the CLI directly:

.. code-block:: bash

    python -m los_estimator --config_file los_estimator/default_config.toml

For full CLI documentation, see :doc:`cli_usage`.

Understanding the Results
-------------------------

The analysis generates comprehensive output in the :file:`results/` directory. Each run creates a timestamped folder containing:

- Performance metrics and comparison tables
- Visualizations of fitted distributions and errors
- Animations showing model evolution over time
- Serialized data for post-processing

For a complete description of all output artifacts, see :doc:`output_format`.


Reloading and Re-Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Results can be reloaded without re-running the entire analysis:

.. code-block:: python

    from los_estimator.estimation_run import LosEstimationRun

    # Load previous results
    run = LosEstimationRun.load_run("results/<run_folder>")
    
    # Generate new visualizations with updated settings
    run.visualize_results()
    run.animate_results()

This is useful for:

- Adjusting figure sizes, colors, or styles
- Creating publication-quality plots
- Experimenting with different visualization layouts
- Generating animations with different frame rates

Simply modify the ``visualization_config`` or ``animation_config`` in your configuration file before reloading.

Next Steps
----------

- Explore the :doc:`cli_usage` for advanced configuration options
- Review :doc:`output_format` for detailed artifact documentation
- Check the :doc:`../apiref/api` for programmatic access to results
