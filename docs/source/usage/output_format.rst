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
            └── metrics_test.csv              # Metric values for test data


