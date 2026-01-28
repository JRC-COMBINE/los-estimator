"""Multi-series fitting for length of stay estimation.

This module provides the main fitting class that orchestrates the fitting
process across multiple time windows and distribution types.
"""

import logging
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from los_estimator.config import ModelConfig
from los_estimator.core import SeriesData
from los_estimator.fitting.los_fitter import (
    calc_its_comp,
    calc_its_convolution,
    fit_compartmental,
    fit_convolution,
)

from .fit_results import MultiSeriesFitResults, SeriesFitResult, SingleFitResult

logger = logging.getLogger("los_estimator")


class MultiSeriesFitter:
    """Main class for fitting LOS models across multiple time series.

    Orchestrates the fitting process across multiple time windows and
    distribution types, managing the optimization process and collecting
    results for analysis.

    Attributes:
        all_fit_results (MultiSeriesFitResults): Container for all fit results.
        series_data (SeriesData): Time series data for fitting.
        model_config (ModelConfig): Configuration for model parameters.
        distributions (list[str]): List of distribution types to fit.
        init_parameters (defaultdict): Initial parameters for each distribution.
        debug_config: Configuration for debugging modes.
    """

    all_fit_results: MultiSeriesFitResults

    def __init__(
        self,
        series_data: SeriesData,
        model_config: ModelConfig,
        distributions: list[str],
        init_parameters: dict[str, list[float]],
    ):
        """Initialize the multi-series fitter.

        Args:
            series_data (SeriesData): Time series data to fit models to.
            model_config (ModelConfig): Configuration for fitting parameters.
            distributions (list[str]): List of distribution types to try.
            init_parameters (dict[str, list[float]]): Initial parameter values.
        """
        self.series_data: SeriesData = series_data
        self.model_config: ModelConfig = model_config
        self._distributions: list[str] = distributions
        self.distributions: list[str] = distributions
        self.all_fit_results: MultiSeriesFitResults = MultiSeriesFitResults()
        self.init_parameters: defaultdict[str, list[float]] = defaultdict(list, init_parameters)
        self.debug_config = None

    def DEBUG_MODE(self, debug_config):
        """Configure debug mode settings.

        Sets up debugging options to reduce computation time during development
        by limiting the number of windows and distributions to test.

        Args:
            debug_config: Debug configuration object with boolean flags.
        """
        dc = debug_config
        self.DEBUG = {
            "ONE_WINDOW": dc.one_window,
            "LESS_WINDOWS": dc.less_windows,
        }

        self.window_data = list(self.series_data)

    def _update_past_kernels(self, fit_result, first_window, w, kernel):
        """Update the rolling kernel matrix with the latest fitted kernel.

        For the first window, the entire `all_kernels` buffer is filled with the
        current kernel. For subsequent windows, only the rows corresponding to the
        current training span are updated.

        Args:
            fit_result (SeriesFitResult): Container accumulating kernels across windows.
            first_window (bool): Whether this is the first processed window.
            w (WindowInfo): Window metadata with `train_start` index.
            kernel (np.ndarray): Fitted kernel for the current window.
        """
        if first_window:
            fit_result.all_kernels[:] = kernel
        else:
            fit_result.all_kernels[w.train_start :] = kernel

    def _find_past_kernels(self, fit_result, first_window, w):
        """Return the previously fitted kernel slice to warm-start fitting.

        If not the first window and iterative kernel fitting is enabled, this
        returns a slice of `all_kernels` covering the current training span. This
        can be passed as a prior to the fitter to improve stability.

        Args:
            fit_result (SeriesFitResult): Aggregated results, including `all_kernels`.
            first_window (bool): Whether this is the first processed window.
            w (WindowInfo): Window metadata with indices for the training span.

        Returns:
            np.ndarray | None: Past kernel slice if available, otherwise None.
        """
        past_kernels = None
        if not first_window and self.model_config.iterative_kernel_fit:
            past_kernels = fit_result.all_kernels[w.train_start : w.train_start + self.model_config.kernel_width]
        return past_kernels

    def fit(self):
        """Fit models across all distributions and time windows."""
        all_fit_results = self.all_fit_results

        # --- Main loop ---
        for distro in self.distributions:
            logger.info(f"Fitting distribution: {distro}")
            all_fit_results[distro] = self.fit_distro(distro)

        all_fit_results.bake()

        for distro, fit_result in all_fit_results.items():
            train_mean = fit_result.train_errors.mean()
            test_mean = fit_result.test_errors.mean()
            logger.info(
                f"{distro[:7]}: Mean Train Error: {float(train_mean):.2f}, Mean Test Error: {float(test_mean):.2f}"
            )
        return self.window_data, all_fit_results

    def fit_distro(self, distro):
        """Fit a single distribution across all sliding windows.

        Runs the appropriate fitter (convolutional or compartmental) for each
        window, tracks failures, updates rolling kernels (for convolutional
        models), and returns an aggregated `SeriesFitResult`.

        Args:
            distro (str): Name of the distribution (e.g., "lognorm", "linear", "compartmental").

        Returns:
            SeriesFitResult: Aggregated results for the given distribution.
        """
        model_config = self.model_config
        series_data = self.series_data

        fit_result = SeriesFitResult(distro)
        fit_result.all_kernels = np.zeros((self.series_data.n_days, self.model_config.kernel_width))

        failed_windows = []
        is_first_window = True

        # compartmental models always uses its own fitter
        for window_id, window_info, train_data, test_data in tqdm(self.window_data):
            w = window_info

            try:
                if distro == "compartmental":
                    result_obj = fit_compartmental(
                        train_data,
                        test_data,
                        initial_guess_comp=[1 / 7, 1, 0],
                        kernel_width=model_config.kernel_width,
                    )
                    y_pred = calc_its_comp(
                        series_data.x_full,
                        *result_obj.distro_params,
                        series_data.y_full[0],
                    )
                else:
                    init_vals = self.init_parameters.get(distro)
                    if self.model_config.reuse_last_parametrization:
                        init_vals = self._find_last_valid_parametrization(fit_result, window_id, init_vals)
                    past_kernels = self._find_past_kernels(fit_result, is_first_window, w)

                    result_obj = fit_convolution(
                        distro,
                        train_data,
                        test_data,
                        self.model_config.kernel_width,
                        distro_init_params=init_vals,
                        past_kernels=past_kernels,
                        error_fun=model_config.error_fun,
                        method=model_config.optimizer,
                    )

                    self._update_past_kernels(fit_result, is_first_window, w, result_obj.kernel)

            except Exception as e:
                logger.error(f"Error fitting {distro} on window {window_id}: {e}")
                result_obj = SingleFitResult.create_failed(distro, train_data, test_data)
                raise e

            if not result_obj.success:
                failed_windows.append(window_id)
            fit_result.append(window_info, result_obj)

            is_first_window = False
        if failed_windows:
            logger.warning(f"Failed to fit {distro} on windows: {failed_windows}")
        fit_result.train_data_reproduction = calc_its_convolution(series_data.x_full, fit_result.all_kernels)

        return fit_result

    def _find_last_valid_parametrization(self, fit_result, window_id, init_vals):
        """Find the most recent successful parameter vector to reuse.

        Iterates backward through previously computed windows to locate the last
        successful fit and reuse its `model_config` as the initialization for the
        current window. If none are found, the provided `init_vals` are returned.

        Args:
            fit_result (SeriesFitResult): Aggregated results containing prior fits.
            window_id (int): Current window index used to limit the search range.
            init_vals (list[float] | None): Fallback initial parameter values.

        Returns:
            list[float] | None: Parameter vector from the last successful window, or `init_vals`.
        """
        for prev in reversed(fit_result[:window_id]):
            if not prev:
                continue
            return prev.distro_params
        return init_vals
