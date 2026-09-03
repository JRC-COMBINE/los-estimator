"""Fit result containers: `SingleFitResult` (one window), `SeriesFitResult`
(one distribution across windows), `MultiSeriesFitResults` (all distributions)."""

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
    r"""Optimizer output for one window/distribution fit.

    covariance/covariance_source: dense Laplace posterior covariance of the
    fitted params (or None if inestimable) and which path produced it.
    train_mse: recomputed independently of `error_fun`, for covariance scaling.
    parameter_samples/acceptance_rate: UQ posterior draws (shape (n_accepted, ndim))
    and the fraction of draws inside the sampling box.
    \*_lower/\*_upper: percentile bands on kernel/train/test predictions.
    uq_past_kernels: the `past_kernels` slice used to build the UQ ensemble
    (only set with `iterative_kernel_fit`).
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
    covariance: Optional[NDArray] = None
    covariance_source: Optional[str] = None
    train_mse: Optional[float] = None
    n_residuals: Optional[int] = None
    parameter_samples: Optional[NDArray] = None
    acceptance_rate: Optional[float] = None
    train_sigma: Optional[float] = None
    test_sigma: Optional[float] = None
    kernel_lower: Optional[NDArray] = None
    kernel_upper: Optional[NDArray] = None
    train_lower: Optional[NDArray] = None
    train_upper: Optional[NDArray] = None
    test_lower: Optional[NDArray] = None
    test_upper: Optional[NDArray] = None
    uq_past_kernels: Optional[NDArray] = None

    def __repr__(self):
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
        """Placeholder result for a failed fit (NaN errors, empty arrays)."""
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
            covariance=None,
            covariance_source=None,
            train_mse=None,
            n_residuals=None,
            parameter_samples=None,
            acceptance_rate=None,
            train_sigma=None,
            test_sigma=None,
            kernel_lower=None,
            kernel_upper=None,
            train_lower=None,
            train_upper=None,
            test_lower=None,
            test_upper=None,
            uq_past_kernels=None,
        )


class SeriesFitResult:
    """Per-window `SingleFitResult` objects for one distribution, plus derived
    arrays (train/test error, rolling kernel matrix, transition rate/delay estimates)."""

    distro: str
    window_infos: list[WindowInfo]
    fit_results: list[SingleFitResult]
    train_errors: NDArray
    test_errors: NDArray
    all_kernels: NDArray
    all_kernel_lower: Optional[NDArray]
    all_kernel_upper: Optional[NDArray]
    all_train_lower: Optional[NDArray]
    all_train_upper: Optional[NDArray]
    all_test_lower: Optional[NDArray]
    all_test_upper: Optional[NDArray]
    transition_rates: NDArray
    transition_delays: NDArray

    def __init__(self, distro):
        self.distro = distro
        self.window_infos = []
        self.fit_results = []
        self.train_errors = None
        self.test_errors = None
        self.all_kernels = None

    def append(self, window_info, fit_result):
        self.window_infos.append(window_info)
        self.fit_results.append(fit_result)

    def bake(self):
        """Compute train/test error arrays, uncertainty bands, and transition
        rate/delay estimates (params[0]/params[1]) from the collected fit_results."""
        self._collect_errors()
        self._collect_bands()
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

    #: Band attributes on `SingleFitResult` aggregated window-wise by `bake`.
    BAND_FIELDS = (
        "kernel_lower",
        "kernel_upper",
        "train_lower",
        "train_upper",
        "test_lower",
        "test_upper",
    )

    def _collect_bands(self):
        """Stack each BAND_FIELDS entry into an `all_<field>` matrix of shape
        (n_windows, band_length), mirroring `all_kernels`. Windows without a band
        get a NaN row; a field absent everywhere is set to None."""
        n_windows = len(self.fit_results)
        for field in self.BAND_FIELDS:
            arrays = [getattr(fr, field, None) if fr is not None else None for fr in self.fit_results]
            present = [a for a in arrays if a is not None]
            if not present:
                setattr(self, f"all_{field}", None)
                continue
            width = max(len(a) for a in present)
            matrix = np.full((n_windows, width), np.nan)
            for i, array in enumerate(arrays):
                if array is not None:
                    matrix[i, : len(array)] = array
            setattr(self, f"all_{field}", matrix)

    def _collect_errors(self):
        """Populate train_errors/test_errors; missing fit results count as inf."""
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
        if isinstance(window_id, slice):
            return self.fit_results[window_id]
        if window_id >= len(self.fit_results):
            raise IndexError(f"Window ID {window_id} out of range for {len(self.fit_results)} windows.")
        return self.fit_results[window_id]

    def __setitem__(self, window_id, value):
        if window_id >= len(self.fit_results):
            raise IndexError(f"Window ID {window_id} out of range for {len(self.fit_results)} windows.")
        self.fit_results[window_id] = value

    def __repr__(self):
        return f"SeriesFitResult(distro={self.distro}, n_windows={len(self.window_infos)}, train_relative_error: {len(self.train_errors)}, test_relative_error: {len(self.test_errors)})"


class MultiSeriesFitResults(OrderedDict[str, SeriesFitResult]):
    """Maps distribution name -> `SeriesFitResult`; builds cross-distro error
    matrices, shape (n_windows, n_distros), and a comparison `summary` DataFrame."""

    results: list[SeriesFitResult]
    distros: list[str]
    n_windows: int
    train_errors_by_distro: NDArray
    test_errors_by_distro: NDArray
    transition_rates_by_distro: NDArray
    transition_delays_by_distro: NDArray
    summary: pd.DataFrame

    def __init__(self, distros=None, *args, **kwargs):
        """If `distros` given, pre-populate with empty `SeriesFitResult`s keyed by name."""
        super().__init__(*args, **kwargs)
        if distros is not None:
            for distro in distros:
                self[distro] = SeriesFitResult(distro)
            self.distros = list(self.keys())
            self.results = list(self.values())

    def bake(self):
        """Bake each SeriesFitResult, build the cross-distro error/transition
        matrices, and generate the summary DataFrame. Returns self."""
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
        """Per-distribution mean/median/quartile train+test loss, plus an
        IQR-outlier-excluded mean."""
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
        return f"MultiSeriesFitResults(distros={self.distros}, n_windows={self.n_windows})"
