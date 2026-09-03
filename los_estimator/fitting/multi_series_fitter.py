"""Orchestrates fitting across all sliding windows x distributions."""

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
from .uncertainty import UncertaintyParams, compute_window_uncertainty

logger = logging.getLogger("los_estimator")


class MultiSeriesFitter:
    """Fits every configured distribution across every sliding window and
    collects the results."""

    all_fit_results: MultiSeriesFitResults

    def __init__(
        self,
        series_data: SeriesData,
        model_config: ModelConfig,
        distributions: list[str],
        init_parameters: dict[str, list[float]],
        uncertainty: "UncertaintyParams | None" = None,
    ):
        self.series_data: SeriesData = series_data
        self.model_config: ModelConfig = model_config
        self._distributions: list[str] = distributions
        self.distributions: list[str] = distributions
        self.all_fit_results: MultiSeriesFitResults = MultiSeriesFitResults()
        self.init_parameters: defaultdict[str, list[float]] = defaultdict(list, init_parameters)
        self.debug_config = None
        self.uncertainty: UncertaintyParams = uncertainty or UncertaintyParams()
        self._rng = np.random.default_rng(self.uncertainty.seed)

    def DEBUG_MODE(self, debug_config):
        dc = debug_config
        self.DEBUG = {
            "ONE_WINDOW": dc.one_window,
            "LESS_WINDOWS": dc.less_windows,
        }

        self.window_data = list(self.series_data)

    def _update_past_kernels(self, fit_result, first_window, w, kernel):
        """Write `kernel` into the rolling `all_kernels` buffer: fill entirely on
        the first window, otherwise only from the current train_start onward."""
        if first_window:
            fit_result.all_kernels[:] = kernel
        else:
            fit_result.all_kernels[w.train_start :] = kernel

    def _find_past_kernels(self, fit_result, first_window, w):
        """Kernel slice over the current training span, for use as a fitting
        prior -- only when not the first window and `iterative_kernel_fit` is on."""
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
        """Fit one distribution across all windows (compartmental uses its own
        fitter; others go through the convolution fitter with rolling kernels)."""
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

                    if self.uncertainty.covers(distro) and result_obj.success:
                        compute_window_uncertainty(
                            result_obj,
                            kernel_width=model_config.kernel_width,
                            params=self.uncertainty,
                            rng=self._rng,
                            past_kernels=past_kernels,
                        )

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
        """Walk backward for the most recent successful fit's params; fall back to `init_vals`."""
        for prev in reversed(fit_result[:window_id]):
            if not prev:
                continue
            return prev.distro_params
        return init_vals
