
Key Configuration Sections
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**data_config**
    - ``icu_file``: Path to ICU occupancy data CSV.
    - ``los_file``: Path to hospital LOS data CSV.
    - ``start_day`` and ``end_day``: Time range for analysis.

**model_config**
    - ``kernel_width``: Width of distribution kernel in days (default: 120).
    - ``train_width``: Width of training window in days (default: 102).
    - ``test_width``: Width of test window in days (default: 21).
    - ``step``: Step size for sliding windows (default: 7).
    - ``distributions``: List of distributions to fit (e.g., ``["lognorm", "gaussian", "linear"]``).
    - ``error_fun``: Error function for optimization (``"mse"``, ``"mae"``, etc.).

**debug_config**
    - ``one_window``: Fit only the first window (bool).
    - ``less_windows``: Reduce windows to ~3 for quick testing (bool).
    - ``less_distros``: Test only linear and compartmental (bool).
    - ``only_linear``: Fit only linear models (bool).

**visualization_config**
    - ``show_figures``: Display plots interactively (bool).
    - ``save_figures``: Save plots to disk (bool).
    - ``figsize``: Figure dimensions as ``[width, height]``.

**animation_config**
    - ``show_figures``: Display animations interactively (bool).
    - ``save_figures``: Save animations as GIFs (bool).