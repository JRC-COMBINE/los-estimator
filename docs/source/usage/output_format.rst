Output Format
=============

This page describes the structure and content of all output artifacts produced by the LoS Estimator.

Directory Structure
-------------------

Results are saved to the ``results/`` directory with the following structure:

::

    results/
    └── <YYYYMMDD_HHMM>_dev_step<N>_train<N>_test<N>_fit_admissions_<error_fn>/
        ├── run.log                           # Detailed run log
        ├── run_configurations.toml           # Configuration snapshot
        ├── model_data/
        │   ├── series_data.pkl               # Loaded time series (binary)
        │   ├── all_fit_results.pkl           # All fit results (binary)
        │   ├── visualization_context.pkl     # Visualization metadata (binary)
        │   └── <distro>_models.csv           # Model parameters per window for each distribution function
        ├── figures/
        │   ├── error_comparison.png          # Bar plot of mean and median test and train errors
        │   ├── prediction_all_distros.png    # All distribution and time point predictions overlaid
        │   ├── prediction_error_all_distros.png    # All distribution and time point predictions overlaid with errors.
        │   ├── prediction_error<distro>_fit.png    # Predictions and errors for a specific distribution at all time points
        │   ├── test_error_boxplot.png
        │   ├── test_error_boxplot_no_outliers.png
        │   ├── test_error_boxplot_no_outliers.png
        │   ├── train_error_boxplot.png
        │   ├── train_error_boxplot_no_outliers.png
        │   └── train_vs_test_error.png
        ├── animation/
        │   ├── <distro>fit_<day>.png         # Individual frames
        │   └── <distro>combined_video.gif    # Combined animation
        └── metrics/
            ├── <metric>_test.png             # Metric for each distribution function at each time point
            ├── metrics_train.csv             # Metric values for training data
            ├── metrics_test.csv              # Metric values for test data
            ├── ci_coverage.csv               # UQ only: per-distribution mean coverage vs nominal
            ├── ci_coverage_detail.csv        # UQ only: per-window coverage and band width
            └── coverage_over_time.png        # UQ only: coverage and band width over time

Uncertainty quantification (``uncertainty_config``)
----------------------------------------------------

When ``uncertainty_config.enabled = true``, a Laplace-approximation posterior is
sampled per window/distribution and reduced to percentile bands (see
``los_estimator/fitting/uncertainty.py``). This changes three parts of the
output above:

- ``model_data/<distro>_models.csv`` gains extra columns after the existing
  ``kernel_<i>`` block: ``param_se_<i>`` (parameter standard errors, i.e. the
  square root of the diagonal of the fitted covariance) and
  ``kernel_lo_<i>`` / ``kernel_hi_<i>`` (the percentile band on the kernel).
  A window without a usable band (failed fit, singular Hessian, too few
  accepted posterior draws, ...) gets NaN in these columns rather than
  corrupting the row. When ``uncertainty_config.enabled = false`` these
  columns are omitted entirely and the CSV layout is unchanged from before
  this feature existed.
- ``figures/prediction_error_<distro>_fit.png`` (and the "all distros"
  variants) shade the train/test occupancy bands behind their prediction
  lines, and ``figures/rolling_kernels_<distro>.png`` shades the kernel band
  behind each rolling kernel. Both are no-ops when a window has no band.
- ``metrics/ci_coverage.csv`` reports the empirical coverage of the test band
  per distribution against ``nominal_coverage`` (e.g. 0.90 for a 5/95 band),
  as two numbers: ``mean_coverage`` (band-only average, i.e. coverage among
  windows that produced a band) and ``mean_coverage_effective`` (the same
  average with every band-less window counted as a miss, so a distribution
  that only bands its easy windows can't look well-calibrated by omission).
  It also reports the mean band width and how many windows carried a band
  out of the total (``n_windows_with_band`` / ``n_windows_total``, and their
  ratio ``band_rate``). ``metrics/ci_coverage_detail.csv`` has one row per
  (distribution, window). ``metrics/coverage_over_time.png`` plots both
  coverage and mean band width against the rolling window index.

Relevant ``uncertainty_config`` fields (see
``los_estimator/config/__init__.py``): ``enabled`` (default ``false``),
``n_samples`` (target accepted posterior draws per window, default 1000),
``confidence_interval`` (nominal lower/upper percentiles, default ``(5, 95)``),
``distributions`` (optional subset; ``None``/omitted means every supported
distribution), ``seed`` (optional RNG seed).
