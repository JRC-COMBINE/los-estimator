Quickstart
==========

This guide provides a quick introduction to the LoS Estimator using a synthetic dataset.

Installation
------------

First, clone the repository and set up a virtual environment:

.. code-block:: bash

    git clone git@git.rwth-aachen.de:jrc-combine/los-estimator.git
    cd los-estimator
    python -m venv .venv
    
    # On Windows
    .\.venv\Scripts\activate
    
    # On Linux/macOS
    source .venv/bin/activate
    
    pip install -r requirements.txt

Synthetic Example
-----------------

The package includes a demonstration script that generates synthetic ICU data and performs LoS estimation:

.. code-block:: bash

    python examples/synthetic_example.py

This script demonstrates the complete workflow:

1. Generates synthetic ICU admission time series
2. Simulates occupancy data using a known lognormal LoS distribution
3. Estimates the LoS distribution from the simulated data
4. Compares estimated distributions against the ground truth
5. Generates visualizations and animations

The configuration is read from :file:`examples/synthetic_example.toml`. You can modify parameters in this file or the script to experiment with different settings.

Understanding the Results
-------------------------

Results are saved to :file:`examples/results/` with the following structure:

**Run Metadata**
    - :file:`run.log` - Complete execution log
    - :file:`run_configurations.toml` - Configuration snapshot for reproducibility

**Performance Metrics**
    - :file:`metrics/` - CSV files and plots showing model performance across windows

**Visualizations**
    - :file:`figures/` - Static plots of fitting results, errors, and comparisons

**Animations**
    - :file:`animation/` - Frame-by-frame visualizations showing model evolution
    - :file:`animation/combined_video.gif` - Animated overview of the fitting process

.. image:: ../img/animation.gif
   :alt: Animation Gif
   :align: center
   :width: 1000px

**Model Data**
    - :file:`model_data/` - Fitted parameters per window (CSV format)
    - :file:`model_data/*.pkl` - Serialized Python objects for later analysis

For detailed output documentation, see :doc:`output_format`.

Next Steps
----------

- Try the :doc:`real_data` example with actual ICU data
- Review :doc:`cli_usage` for command-line options
- Explore :doc:`input` to understand data requirements
- Check the :doc:`../apiref/api` for programmatic usage
