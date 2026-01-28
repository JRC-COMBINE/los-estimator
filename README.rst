LoS Estimator
=============

A tool for estimating Length of Stay (LoS) distributions in healthcare settings using deconvolution techniques.

Overview
--------

The LoS Estimator derives patient length of stay distributions from ICU admission and occupancy time series data. It employs statistical fitting techniques to identify the underlying probability distribution that best explains observed occupancy patterns, enabling data-driven resource planning and capacity management.

.. note::
    This work builds on the methodology described in:

    A. Schuppert, S. Theisen, P. Fränkel, S. Weber-Carstens, C. Karagiannidis.

    Bundesweites Belastungsmodell für Intensivstationen durch COVID-19.
    (English: Nationwide exposure model for COVID-19 intensive care unit admission)

    
    doi: https://doi.org/10.1007/s00063-021-00791-7


Key Features
------------

- **Multiple Distribution Support:** Fit lognormal, gamma, Gaussian, exponential, and compartmental models
- **Rolling Window Analysis:** Track temporal changes in LoS distributions over time
- **Automated Model Selection:** Compare distributions and identify the best-fitting model
- **Rich Visualizations:** Generate plots and animations of fitting results
- **Flexible Configuration:** TOML-based configuration with command-line overrides
- **Real-World Data Ready:** Includes preprocessing tools for RKI COVID-19 ICU data

Methodology
-----------

The estimator is built on the following principles:

**Convolution Model**
    We assume that individual patient length of stay (LoS) follows a probability distribution (e.g., lognormal, gamma). By convolving the admission time series with discharge probabilities derived from the LoS distribution, we can model expected occupancy over time:

    .. math::

        \text{Occupancy}(t) = \sum_{\tau=0}^{t} \text{Admissions}(t-\tau) \cdot P(\text{LoS} > \tau)

**Deconvolution Problem**
    Given observed admissions and occupancy, estimating the LoS distribution can be described as the inverse problem of fitting distribution parameters to minimize prediction error.

**Temporal Dynamics**
    LoS distributions may shift due to treatment protocol changes, patient demographics, or disease characteristics. The estimator uses a rolling window approach to track these changes, fitting distributions on overlapping time windows.

The animation below illustrates the rolling window training process:

.. image:: img/animation.gif
   :alt: Rolling Window LoS Estimation Animation
   :align: center
   :width: 800px

*Figure: Evolution of fitted LoS distributions across time windows*

Quick Start
-----------

**Installation**

.. code-block:: bash

    git clone git@git.rwth-aachen.de:jrc-combine/los-estimator.git
    cd los-estimator
    python -m venv .venv
    
    # On Windows
    .\.venv\Scripts\activate
    
    # On Linux/macOS
    source .venv/bin/activate
    
    pip install -r requirements.txt

**Run Synthetic Example**

.. code-block:: bash

    python examples/synthetic_example.py

**Run with Real Data**

.. code-block:: bash

    python -m los_estimator --config_file los_estimator/default_config.toml

Documentation
-------------

Full documentation is available at: `https://los-estimator.readthedocs.io/ <https://los-estimator.readthedocs.io/>`_

- `Quickstart Guide <docs/source/usage/quickstart.rst>`_
- `CLI Reference <docs/source/usage/cli_usage.rst>`_
- `Input Format <docs/source/usage/input.rst>`_
- `Output Format <docs/source/usage/output_format.rst>`_
- `API Reference <docs/source/apiref/api.rst>`_

Project Structure
-----------------

::

    los-estimator/
    ├── los_estimator/          # Main package
    │   ├── cli/                # Command-line interface
    │   ├── config/             # Configuration management
    │   ├── core/               # Core data structures
    │   ├── data/               # Data loading and preprocessing
    │   ├── evaluation/         # Model evaluation
    │   ├── fitting/            # Distribution fitting algorithms
    │   └── visualization/      # Plotting and animation
    ├── examples/               # Example scripts and data
    ├── docs/                   # Sphinx documentation
    ├── tests/                  # Unit and integration tests
    └── results/                # Output directory (created at runtime)


Contact
-------

For questions, issues, or contributions:

- **Issues:** `GitLab Issue Tracker <https://git.rwth-aachen.de/jrc-combine/los-estimator/-/issues>`_
- **Author:** Younes Müller
- **Institution:** RWTH Aachen University


Links
-----

    Documentation: `https://los-estimator.readthedocs.io/ <https://los-estimator.readthedocs.io/>`_

    Source: `https://github.com/Younesmueller/los-estimator <https://github.com/Younesmueller/los-estimator>`_

    Issue tracker: `https://github.com/Younesmueller/los-estimator/issues <https://github.com/Younesmueller/los-estimator/issues>`_
