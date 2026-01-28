"""Containers for fit results and cross-distribution summaries.

This module defines lightweight containers used throughout the fitting pipeline:

- `SingleFitResult`: Result of fitting a single window for one distribution.
- `SeriesFitResult`: Aggregates window-wise `SingleFitResult` for a distribution
  and computes derived arrays and metrics.
- `MultiSeriesFitResults`: Aggregates multiple `SeriesFitResult` instances across
  distributions and produces a comparison-ready summary.
"""

from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from los_estimator.config import ModelConfig
from los_estimator.core import WindowInfo
from numpy.typing import NDArray


@dataclass
class SingleFitResult:
    """Result of a single-window fit for one distribution.

    Instances capture artifacts produced by the optimizer for a specific window,
    including errors, predictions, and the fitted kernel. Relative error arrays
    and fitted curve can be attached if computed downstream.

    Attributes:
        distro (str): Distribution name (e.g., "lognorm", "gaussian").
        train_data (object): Training data used for the fit.
        test_data (object): Held-out data for evaluation.
        success (bool): Whether the optimizer reported success.
        minimization_result (dict): Raw optimizer result object.
        train_error (NDArray): Scalar training error for the window.
        test_error (NDArray): Scalar test error for the window.
        kernel (NDArray): Fitted kernel array for this window.
        model_config (ModelConfig): Model configuration snapshot used during fitting.
        train_prediction (NDArray | None): Optional model predictions on the training window.
        test_prediction (NDArray | None): Optional model predictions on the test window.
        rel_train_error (NDArray | None): Optional element-wise relative training error array.
        rel_test_error (NDArray | None): Optional element-wise relative test error array.
        curve (NDArray | None): Optional fitted curve points for visualization.
    """

    distro: str
    train_data: object
    test_data: object
    success: bool
    minimization_result: dict
    train_error: NDArray
    test_error: NDArray
    kernel: NDArray
    distro_params: NDArray
    train_prediction: Optional[NDArray] = None
    test_prediction: Optional[NDArray] = None
    rel_train_error: Optional[NDArray] = None
    rel_test_error: Optional[NDArray] = None
    curve: Optional[NDArray] = None

    def __repr__(self):
        """Return a concise summary of the single fit result.

        Returns:
            str: Readable summary string including key fields and shapes.
        """
        # return a string with all variables
        if self is None:
            return None
        return (
            f"SingleFitResult(distro={self.distro}, "
            f"success={self.success}, "
            f"train_error={self.train_error}, "
            f"test_error={self.test_error}, "
            f"rel_train_error={self.rel_train_error}, "
            f"rel_test_error={self.rel_test_error}, "
            f"kernel={self.kernel.shape}, "
            f"distro_params={self.distro_params})"
        )

    @classmethod
    def create_failed(cls, distro, train_data, test_data):
        """Create a SingleFitResult representing a failed fit.

        Args:
            distro (str): Distribution name.
            train_data (object): Training data used for the fit.
            test_data (object): Held-out data for evaluation.

        Returns:
            SingleFitResult: Instance indicating failure with NaN errors.
        """
        return cls(
            distro=distro,
            train_data=train_data,
            test_data=test_data,
            success=False,
            minimization_result={},
            train_error=np.nan,
            test_error=np.nan,
            kernel=np.array([]),
            distro_params=np.array([]),
        )


class SeriesFitResult:
    """Aggregate fit results across windows for a single distribution.

    A `SeriesFitResult` collects `SingleFitResult` objects produced for each
    sliding window of a specific distribution and computes convenient arrays
    like `train_errors`, `test_errors`, and derived transition metrics.

    Attributes:
        distro (str): Distribution name for this series.
        window_infos (list[WindowInfo]): Metadata for each processed window.
        fit_results (list[SingleFitResult]): Per-window fit result objects.
        train_errors (NDArray): Array of training errors across windows.
        test_errors (NDArray): Array of test errors across windows.
        all_kernels (NDArray): Rolling kernel matrix (rows = day, cols = kernel width).
        transition_rates (NDArray): Transition rate estimates from model configs.
        transition_delays (NDArray): Transition delay estimates from model configs.
    """

    distro: str
    window_infos: list[WindowInfo]
    fit_results: list[SingleFitResult]
    train_errors: NDArray
    test_errors: NDArray
    all_kernels: NDArray
    transition_rates: NDArray
    transition_delays: NDArray

    def __init__(self, distro):
        """Initialize the series container for a given distribution.

        Args:
            distro (str): Distribution name (e.g., "lognorm", "gaussian").
        """
        self.distro = distro
        self.window_infos = []
        self.fit_results = []
        self.train_errors = None
        self.test_errors = None
        self.all_kernels = None

    def append(self, window_info, fit_result):
        """Append a window and its fit result to the series.

        Args:
            window_info (WindowInfo): Metadata about the processed window.
            fit_result (SingleFitResult): The fit result for the window.
        """
        self.window_infos.append(window_info)
        self.fit_results.append(fit_result)

    def bake(self):
        """Finalize the series by computing derived arrays and metrics.

        This method aggregates individual window results and computes:

        - Training and test error arrays across all windows
        - Success/failure counts for optimizer convergence
        - Transition rate and delay estimates from model parameters
        """
        self._collect_errors()
        self.transition_rates = np.array(
            [
                (fr.distro_params[0] if ((fr is not None) and len(fr.distro_params) > 0) else np.nan)
                for fr in self.fit_results
            ]
        )
        self.transition_delays = np.array(
            [
                (fr.distro_params[1] if ((fr is not None) and len(fr.distro_params) > 1) else np.nan)
                for fr in self.fit_results
            ]
        )

    def _collect_errors(self):
        """Collect window-wise training and testing errors into arrays.

        Populates `train_errors` and `test_errors` by iterating over `fit_results`.
        Missing fit results are treated as infinite error.
        """
        self.errors_collected = True
        train_err = np.empty(len(self.fit_results))
        test_err = np.empty(len(self.fit_results))
        for i, fr in enumerate(self.fit_results):
            if fr is None:
                train_err[i] = np.inf
                test_err[i] = np.inf
                continue
            train_err[i] = fr.train_error
            test_err[i] = fr.test_error
        self.train_errors = train_err
        self.test_errors = test_err

    def __getitem__(self, window_id):
        """Return fit result(s) for a given window index or slice.

        Args:
            window_id (int | slice): Window index or slice.

        Returns:
            SingleFitResult | list[SingleFitResult]: The corresponding result(s).

        Raises:
            IndexError: If `window_id` is an integer out of range.
        """
        if isinstance(window_id, slice):
            return self.fit_results[window_id]
        if window_id >= len(self.fit_results):
            raise IndexError(f"Window ID {window_id} out of range for {len(self.fit_results)} windows.")
        return self.fit_results[window_id]

    def __setitem__(self, window_id, value):
        """Replace the fit result at a given window index.

        Args:
            window_id (int): Window index to replace.
            value (SingleFitResult): New fit result.

        Raises:
            IndexError: If `window_id` is out of range.
        """
        if window_id >= len(self.fit_results):
            raise IndexError(f"Window ID {window_id} out of range for {len(self.fit_results)} windows.")
        self.fit_results[window_id] = value

    def __repr__(self):
        """Return a concise summary of the series fit result.

        Returns:
            str: Summary including distro, window count, and error array sizes.
        """
        return f"SeriesFitResult(distro={self.distro}, n_windows={len(self.window_infos)}, train_relative_error: {len(self.train_errors)}, test_relative_error: {len(self.test_errors)})"


class MultiSeriesFitResults(OrderedDict[str, SeriesFitResult]):
    """Aggregate results for multiple distributions and build summaries.

    This container maps distribution names to their `SeriesFitResult` instances,
    and offers helpers to compute cross-distro arrays (e.g., error matrices)
    and a comparison `summary` DataFrame.

    Attributes:
        results (list[SeriesFitResult]): List of SeriesFitResult instances for each distribution.
        distros (list[str]): List of distribution names corresponding to the results.
        n_windows (int): Number of fitting windows across all distributions.
        train_errors_by_distro (NDArray): 2D array of training errors with shape (n_windows, n_distros).
        test_errors_by_distro (NDArray): 2D array of test errors with shape (n_windows, n_distros).
        transition_rates_by_distro (NDArray): 2D array of transition rates with shape (n_windows, n_distros).
        transition_delays_by_distro (NDArray): 2D array of transition delays with shape (n_windows, n_distros).
        summary (pd.DataFrame): Comparison DataFrame with error statistics and robustness metrics per distribution.

    """

    results: list[SeriesFitResult]
    distros: list[str]
    n_windows: int
    train_errors_by_distro: NDArray
    test_errors_by_distro: NDArray
    transition_rates_by_distro: NDArray
    transition_delays_by_distro: NDArray
    summary: pd.DataFrame

    def __init__(self, distros=None, *args, **kwargs):
        """Initialize the multi-series container.

        If `distros` is provided, the container is pre-populated with empty
        `SeriesFitResult` instances keyed by the distribution names.

        Args:
            distros (list[str] | None): Optional list of distribution names.
            *args: Passthrough positional args for `OrderedDict`.
            **kwargs: Passthrough keyword args for `OrderedDict`.
        """
        super().__init__(*args, **kwargs)
        if distros is not None:
            for distro in distros:
                self[distro] = SeriesFitResult(distro)
            self.distros = list(self.keys())
            self.results = list(self.values())

    def bake(self):
        """Finalize all series and compute cross-distribution arrays.

        Processes all distribution results and creates comparison matrices:

        - Calls bake() on each SeriesFitResult
        - Builds train/test error matrices (windows × distributions)
        - Computes transition rate and delay matrices
        - Generates summary DataFrame with statistics and robustness metrics

        Returns:
            MultiSeriesFitResults: Self, for method chaining.
        """
        self.distros = list(self.keys())
        self.results = list(self.values())

        for distro, fit_result in self.items():
            fit_result.bake()
        self.n_windows = len(self.results[0].fit_results) if self.results else 0
        self.train_errors_by_distro = np.array([fr.train_errors for fr in self.results]).T
        self.test_errors_by_distro = np.array([fr.test_errors for fr in self.results]).T
        self.transition_rates_by_distro = np.array([fr.transition_rates for fr in self.results]).T
        self.transition_delays_by_distro = np.array([fr.transition_delays for fr in self.results]).T
        self.n_windows = len(self.results[0].fit_results) if self.results else 0

        self._make_summary()
        return self

    def _make_summary(self):
        """Build a comparison DataFrame summarizing error statistics per distribution.

        Computes comprehensive summary statistics including:

        - Mean and median train/test loss
        - Upper and lower quartiles for training loss
        - Robust mean estimates excluding IQR-based outliers
        """
        df_train = pd.DataFrame(self.train_errors_by_distro, columns=self.distros)
        df_test = pd.DataFrame(self.test_errors_by_distro, columns=self.distros)

        summary = pd.DataFrame(index=self.distros)

        summary["Mean Loss Train"] = df_train.replace(np.inf, np.nan).mean()
        summary["Median Loss Train"] = df_train.replace(np.inf, np.nan).median()
        summary["Upper Quartile Train"] = df_train.quantile(0.75)
        summary["Lower Quartile Train"] = df_train.quantile(0.25)

        summary["Mean Loss Test"] = df_test.replace(np.inf, np.nan).mean()
        summary["Median Loss Test"] = df_test.replace(np.inf, np.nan).median()

        def remove_outliers(df, col):
            summary[col] = np.nan
            for distro in self.distros:
                Q1, Q3 = df[distro].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                # filter out outliers
                mask = (df[distro] < (Q1 - 1.5 * IQR)) | (df[distro] > (Q3 + 1.5 * IQR))
                summary.at[distro, col] = df[distro][~mask].mean()

        remove_outliers(df_test, "Mean Loss Test (no outliers)")
        remove_outliers(df_train, "Mean Loss Train (no outliers)")

        self.summary = summary

    def __repr__(self):
        """Return a concise summary of the multi-series results.

        Returns:
            str: Summary including distribution list and window count.
        """
        return f"MultiSeriesFitResults(distros={self.distros}, n_windows={self.n_windows})"
