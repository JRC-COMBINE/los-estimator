Input Format
============

This page describes the required input formats for running the LoS Estimator.

Configuration Files
-------------------

Configuration is managed via TOML files. The default configuration at :file:`los_estimator/default_config.toml` provides all necessary settings.

.. include:: _key_configs.rst

Data Files
----------

ICU Data (Required)
^^^^^^^^^^^^^^^^^^^

The primary input is a CSV file containing ICU admission and occupancy time series with the following columns:

- ``date`` - Date in YYYY-MM-DD format
- ``admissions`` - Number of new ICU admissions on that date (integer)
- ``occupancy`` - Total ICU beds occupied on that date (integer)

**Example:**

.. code-block:: text

    date,admissions,occupancy
    2020-01-01,5,20
    2020-01-02,3,22
    2020-01-03,7,25
    ...

**File Path:** Specify the path to this file in your configuration as ``data_config.icu_file``.

Initial Parameters (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can provide initial parameter values for optimization to improve convergence. The file should be CSV formatted:

- ``distro`` - Distribution name (e.g., "lognorm", "gamma", "gaussian")
- ``params`` - Space-separated parameter values in brackets, e.g., ``[2.5 1.0 1.2]``

**Example:**

.. code-block:: text

    distro,params
    lognorm,[2.5 0.8 1.0]
    gamma,[3.0 0.5 1.0]

**File Path:** Specify as ``data_config.init_params_file`` in your configuration.

Sample LoS Distribution (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For validation purposes, you can provide a known LoS distribution (typically for synthetic data). The file should be CSV formatted:

- ``day`` - Length of stay in days (integer, 0 to max_los)
- ``probability`` - Discharge probability for that day (float, 0-1)

**Example:**

.. code-block:: text

    day,probability
    0,0.05
    1,0.10
    2,0.15
    ...

**File Path:** Specify as ``data_config.los_file`` in your configuration.

**Note:** The package includes a sample distribution file for the synthetic example.



