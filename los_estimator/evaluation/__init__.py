"""Evaluation utilities for LOS estimator.

This module provides:
- EvaluationResult: container for metric arrays and helpers to access them.
- WindowDataPackage: packages windowed train/test predictions and true values.
- Evaluator: computes metrics over windowed results and can save them to CSV.


"""

from typing import Any, Callable, Iterator, List, Optional, Tuple
import pandas as pd
import numpy as np
from numpy.typing import NDArray

from los_estimator.core import SeriesData
from los_estimator.fitting.errors import ErrorFunctions
from los_estimator.fitting.fit_results import MultiSeriesFitResults


class WindowDataPackage:
    """Pack windowed train/test predictions and ground-truth into a 2D object array for evaluation.

    The data attribute is an array shaped (n_distros, n_windows) with tuples:
    (y_true_train, y_pred_train, y_true_test, y_pred_test, x_train, x_test, window_info)
    """

    def __init__(
        self, all_fit_results: MultiSeriesFitResults, series_data: SeriesData
    ) -> None:
        self.all_fit_results: MultiSeriesFitResults = all_fit_results
        self.series_data: SeriesData = series_data
        self.data: Optional[NDArray[Any]] = None
        self.n_distros: int = 0
        self.n_windows: int = 0
        self.build_package()

    def iterate_index(self) -> Iterator[Tuple[int, int]]:
        """Yield (i_distro, i_window) tuples."""
        for i_distro in range(self.n_distros):
            for i_window in range(self.n_windows):
                yield i_distro, i_window

    def build_package(self) -> None:
        """Construct the data array from all_fit_results and series_data."""
        self.n_distros = len(self.all_fit_results)
        self.n_windows = self.all_fit_results.n_windows
        data: NDArray[Any] = np.empty((self.n_distros, self.n_windows), dtype=object)

        for i_distro, fit_result in enumerate(self.all_fit_results.values()):
            for i_window, (single_fit_result, w) in enumerate(
                zip(fit_result.fit_results, fit_result.window_infos)
            ):
                y_pred_train = single_fit_result.train_prediction[w.kernel_width :]
                y_pred_test = single_fit_result.test_prediction[w.kernel_width :]

                y_true_train = self.series_data.y_full[
                    w.training_prediction_start : w.train_end
                ]
                y_true_test = self.series_data.y_full[w.test_start : w.test_end]
                x_train = np.arange(w.training_prediction_start, w.train_end)
                x_test = np.arange(w.test_start, w.test_end)

                data[i_distro][i_window] = (
                    y_true_train,
                    y_pred_train,
                    y_true_test,
                    y_pred_test,
                    x_train,
                    x_test,
                    w,
                )

        self.data = data


class EvaluationResult:
    """Container for evaluation metric results.

    Attributes:
        train (Optional[NDArray[Any]]): 3D array (n_distros, n_windows, n_metrics) for training metrics.
        test (Optional[NDArray[Any]]): 3D array (n_distros, n_windows, n_metrics) for test metrics.
        distros (List[str]): ordered list of distribution names.
        metric_names (List[str]): ordered list of metric names.
    """

    def __init__(
        self,
        distros: List[str],
        metric_names: List[str],
        train: Optional[NDArray[Any]] = None,
        test: Optional[NDArray[Any]] = None,
    ) -> None:
        self.train: Optional[NDArray[Any]] = train
        self.test: Optional[NDArray[Any]] = test
        self.distros: List[str] = distros
        self.metric_names: List[str] = metric_names

    def get_dfs(self) -> pd.DataFrame:
        """Return a DataFrame with train/test mean and median per distribution and metric."""
        if self.train is None or self.test is None:
            raise RuntimeError("Metrics not available - call calculate_metrics() first")

        # Use precomputed summaries if present, otherwise compute from arrays
        train_mean = getattr(self, "train_mean", None)
        if train_mean is None:
            train_mean = self.train.mean(axis=1)
        test_mean = getattr(self, "test_mean", None)
        if test_mean is None:
            test_mean = self.test.mean(axis=1)

        train_median = getattr(self, "train_median", None)
        if train_median is None:
            train_median = np.median(self.train, axis=1)
        test_median = getattr(self, "test_median", None)
        if test_median is None:
            test_median = np.median(self.test, axis=1)

        rows = []
        for i_distro, distro in enumerate(self.distros):
            for i_metric, metric in enumerate(self.metric_names):
                rows.append(
                    {
                        "distribution": distro,
                        "metric": metric,
                        "train_mean": float(train_mean[i_distro, i_metric]),
                        "train_median": float(train_median[i_distro, i_metric]),
                        "test_mean": float(test_mean[i_distro, i_metric]),
                        "test_median": float(test_median[i_distro, i_metric]),
                    }
                )
        return pd.DataFrame(rows)


class Evaluator:
    """Compute metrics over windowed predictions for multiple distributions.

    Parameters:
        all_fit_results (MultiSeriesFitResults): Results container for multiple series fits.
        series_data (SeriesData): SeriesData instance with ground-truth and window definitions.
    """

    def __init__(
        self,
        all_fit_results: MultiSeriesFitResults,
        series_data: SeriesData,
    ) -> None:
        self.all_fit_results: MultiSeriesFitResults = all_fit_results
        self.series_data: SeriesData = series_data
        metrics = ErrorFunctions.errors.items()
        self.metric_names: List[str] = [m[0] for m in metrics]
        self.metric_functions: List[Callable[[NDArray[Any], NDArray[Any]], float]] = [
            m[1] for m in metrics
        ]
        self.result: Optional[EvaluationResult] = EvaluationResult(
            distros=list(all_fit_results.keys()), metric_names=self.metric_names
        )
        self.window_data_package = WindowDataPackage(
            self.all_fit_results, self.series_data
        )
        # link package to result for convenience
        self.result.window_data_package = self.window_data_package

    def calculate_metrics(self) -> EvaluationResult:
        """Compute the configured metrics over all distributions and windows.

        Returns:
            EvaluationResult containing train and test metric arrays.
        """
        n_distros = self.window_data_package.n_distros
        n_windows = self.window_data_package.n_windows
        res_train: NDArray[Any] = np.zeros(
            (n_distros, n_windows, len(self.metric_names)), dtype=float
        )
        res_test: NDArray[Any] = np.zeros(
            (n_distros, n_windows, len(self.metric_names)), dtype=float
        )
        assert self.window_data_package.data is not None

        for i_distro, i_window in self.window_data_package.iterate_index():
            y_true_train, y_pred_train, y_true_test, y_pred_test, *_ = (
                self.window_data_package.data[i_distro][i_window]
            )

            for i_metric, metric_func in enumerate(self.metric_functions):
                res_train[i_distro, i_window, i_metric] = float(
                    metric_func(y_true_train, y_pred_train)
                )
                res_test[i_distro, i_window, i_metric] = float(
                    metric_func(y_true_test, y_pred_test)
                )

        if self.result is None:
            # shouldn't happen, but guard for mypy/typing completeness
            self.result = EvaluationResult(
                distros=list(self.all_fit_results.keys()),
                metric_names=self.metric_names,
            )

        self.result.train = res_train
        self.result.test = res_test

        self.result.train_mean = res_train.mean(axis=1)
        self.result.test_mean = res_test.mean(axis=1)
        self.result.train_median = np.median(res_train, axis=1)
        self.result.test_median = np.median(res_test, axis=1)
        return self.result

    def save_result(self, path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Persist metrics to CSV files (metrics_train.csv and metrics_test.csv).

        Returns:
            (df_train, df_test) DataFrames written to disk.
        """
        if self.result is None or self.result.train is None or self.result.test is None:
            raise RuntimeError(
                "Metrics have not been calculated. Call calculate_metrics() first."
            )

        distros = self.result.distros
        metrics = self.metric_names
        # windows should be provided by SeriesData
        windows = getattr(self.series_data, "windows", None)
        if windows is None:
            raise AttributeError(
                "series_data must expose a 'windows' attribute for save_result()"
            )

        arr_train = self.result.train
        arr_test = self.result.test
        n_dists, n_days, n_metrics = arr_train.shape

        df_train = (
            pd.DataFrame(
                arr_train.reshape(n_dists * n_days, n_metrics), columns=metrics
            )
            .assign(
                distribution=np.repeat(distros, n_days), day=np.tile(windows, n_dists)
            )
            .melt(
                id_vars=["distribution", "day"], var_name="metric", value_name="value"
            )
        )
        df_test = (
            pd.DataFrame(arr_test.reshape(n_dists * n_days, n_metrics), columns=metrics)
            .assign(
                distribution=np.repeat(distros, n_days), day=np.tile(windows, n_dists)
            )
            .melt(
                id_vars=["distribution", "day"], var_name="metric", value_name="value"
            )
        )

        df_train = df_train[["metric", "day", "distribution", "value"]]
        df_test = df_test[["metric", "day", "distribution", "value"]]

        df_train.to_csv(path + "/metrics_train.csv", index=False)
        df_test.to_csv(path + "/metrics_test.csv", index=False)
        return df_train, df_test
