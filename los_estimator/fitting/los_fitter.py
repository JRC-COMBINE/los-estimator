"""Length of Stay fitting algorithms and utilities.

This module provides functions for fitting convolution-based models to hospital
length of stay data using various distribution types and optimization methods.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from typing import Callable, Optional, Tuple, List

from los_estimator.fitting.models.compartmental_model import calc_its_comp
from los_estimator.fitting.models.convolutional_model import calc_its_convolution

from .distributions import Distributions
from .errors import ErrorFunctions
from .fit_results import SingleFitResult


def combine_past_kernel(past_kernels: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Combine past kernels with a new kernel.

    Args:
        past_kernels (np.ndarray): Previously fitted kernels.
        kernel (np.ndarray): New kernel to add.

    Returns:
        np.ndarray: Combined kernels stacked vertically.
    """
    return np.vstack([*past_kernels, kernel])


def get_objective_convolution(
    distro: str, kernel_width: int, error_fun: Callable[[np.ndarray, np.ndarray], float]
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]], float]:
    """Create an objective function for convolution-based fitting.

    Args:
        distro (str): Distribution type for the kernel.
        kernel_width (int): Width of the kernel in days.
        error_fun (callable): Error function to minimize.

    Returns:
        callable: Objective function that takes model parameters and data.
    """

    def objective_function(
        model_config: np.ndarray,
        inc: np.ndarray,
        icu: np.ndarray,
        past_kernels: Optional[np.ndarray] = None,
        return_prediction=False,
    ) -> float:
        kernel = Distributions.generate_kernel(distro, model_config, kernel_width)
        if past_kernels is not None:
            kernel = combine_past_kernel(past_kernels, kernel)

        observed = calc_its_convolution(inc, kernel)
        res = error_fun(icu[kernel_width:], observed[kernel_width:])
        if return_prediction:
            return res, observed
        return res

    return objective_function


def initialize_distro_parameters(
    distro: str,
    distro_boundaries: Optional[List[Tuple[Optional[float], Optional[float]]]],
    distro_init_params: Optional[List[float]],
) -> Tuple[List[Tuple[Optional[float], Optional[float]]], List[float]]:
    """Initialize distribution parameters and boundaries for optimization.

    Sets up default boundaries and initial parameters for a given distribution
    if not provided by the user.

    Args:
        distro (str): Distribution type name.
        distro_boundaries (list, optional): Parameter boundaries for optimization.
        distro_init_params (list, optional): Initial parameter values.

    Returns:
        tuple: (distro_boundaries, distro_init_params) with defaults applied.
    """

    if distro_boundaries is None:
        stretch_factor_bounds = [(None, None)]
        distro_boundaries = Distributions[distro].boundaries + stretch_factor_bounds

    if distro_init_params is None or len(distro_init_params) == 0:
        stretching_init = 1
        distro_init_params = Distributions[distro].init_values + [stretching_init]

    return distro_boundaries, distro_init_params


def fit_convolution(
    distro: str,
    train_data: Tuple[np.ndarray, np.ndarray],
    test_data: Tuple[np.ndarray, np.ndarray],
    kernel_width: int,
    distro_boundaries: Optional[List[Tuple[Optional[float], Optional[float]]]] = None,
    distro_init_params: Optional[List[float]] = None,
    past_kernels: Optional[np.ndarray] = None,
    method: str = "L-BFGS-B",
    error_fun: str = "mse",
) -> SingleFitResult:
    """Fit a convolution-based model to length of stay data.

    Performs optimization to find the best parameters for a specified distribution
    that minimizes the error between predicted and observed ICU occupancy.

    Args:
        distro (str): Distribution type for the kernel (e.g., 'gamma', 'lognormal').
        train_data (tuple): (x_train, y_train) training data arrays.
        test_data (tuple): (x_test, y_test) test data arrays.
        kernel_width (int): Width of the distribution kernel in days.
        distro_boundaries (list, optional): Parameter bounds for optimization.
        distro_init_params (list, optional): Initial parameter values.
        past_kernels (np.ndarray, optional): Previously fitted kernels to combine.
        method (str, optional): Optimization method. Defaults to "L-BFGS-B".
        error_fun (str, optional): Error function name. Defaults to "mse".

    Returns:
        SingleFitResult: Object containing fit results, parameters, and predictions.
    """

    error_fun = ErrorFunctions[error_fun]

    distro_boundaries, distro_init_params = initialize_distro_parameters(distro, distro_boundaries, distro_init_params)

    obj_fun = get_objective_convolution(distro, kernel_width, error_fun)

    args = (
        *train_data,
        past_kernels,
    )

    result = minimize(
        obj_fun,
        x0=distro_init_params,
        args=args,
        bounds=distro_boundaries,
        method=method,
    )

    distro_params = result.x

    fitted_kernel = Distributions.generate_kernel(distro, distro_params, kernel_width)

    train_err, train_prediction = obj_fun(distro_params, *train_data, past_kernels, return_prediction=True)
    test_err, test_prediction = obj_fun(distro_params, *test_data, past_kernels, return_prediction=True)

    rel_train_error = train_prediction[kernel_width:] / train_data[1][kernel_width:]
    rel_test_error = test_prediction[kernel_width:] / test_data[1][kernel_width:]

    fit_results = SingleFitResult(
        distro=distro,
        train_data=train_data,
        test_data=test_data,
        success=result.success,
        minimization_result=result,
        train_error=train_err,
        test_error=test_err,
        kernel=fitted_kernel,
        train_prediction=train_prediction,
        test_prediction=test_prediction,
        distro_params=distro_params,
        rel_train_error=rel_train_error,
        rel_test_error=rel_test_error,
    )

    return fit_results


def objective_compartemental(
    error_fun: Callable[[np.ndarray, np.ndarray], float],
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, int], float]:
    """Create an objective function for compartmental model fitting.

    Args:
        error_fun (callable): Error function to minimize.

    Returns:
        callable: Objective function for compartmental model optimization.
    """

    def objective_function(model_config: np.ndarray, inc: np.ndarray, icu: np.ndarray, kernel_width: int) -> float:
        discharge_rate, transition_rate, delay = model_config
        pred = calc_its_comp(inc, discharge_rate, transition_rate, delay, init=icu[0])
        return error_fun(pred[kernel_width : len(icu)], icu[kernel_width:])

    return objective_function


def fit_compartmental(
    train_data: Tuple[np.ndarray, np.ndarray],
    test_data: Tuple[np.ndarray, np.ndarray],
    initial_guess_comp: List[float],
    kernel_width: int,
    method: str = "TNC",
    error_fun: str = "mse",
) -> SingleFitResult:
    """Fit the compartmental model to a train/test split.

    Runs bounded optimization over the compartmental parameters and returns a
    `SingleFitResult` with errors, predictions, and the fitted configuration.

    Args:
        train_data (tuple[np.ndarray, np.ndarray]): (x_train, y_train) arrays.
        test_data (tuple[np.ndarray, np.ndarray]): (x_test, y_test) arrays.
        initial_guess_comp (list[float]): Initial values for the compartmental parameters.
        kernel_width (int): Window width used when slicing evaluation arrays.
        method (str, optional): Optimization method. Defaults to "TNC".
        error_fun (str, optional): Error function name. Defaults to "mse".

    Returns:
        SingleFitResult: Fit result including errors, predictions, and model config.
    """
    x_train, y_train = train_data
    x_test, y_test = test_data

    error_fun = ErrorFunctions[error_fun]
    obj_fun = objective_compartemental(error_fun)

    result = minimize(
        obj_fun,
        initial_guess_comp,
        args=(x_train, y_train, kernel_width),
        method=method,
        bounds=[(0, 1), (1, 1), (0, 0)],
    )
    train_prediction = calc_its_comp(x_train, *result.x, y_train[0])
    test_prediction = calc_its_comp(x_test, *result.x, y_test[0])

    train_err = obj_fun(result.x, x_train, y_train, kernel_width)
    test_err = obj_fun(result.x, x_test, y_test, kernel_width)

    rel_train_error = train_prediction / train_data[1]
    rel_test_error = test_prediction / test_data[1]

    result_obj = SingleFitResult(
        distro="compartmental",
        train_data=x_train,
        test_data=x_test,
        success=result.success,
        minimization_result=result,
        train_error=train_err,
        test_error=test_err,
        kernel=np.zeros(1),
        train_prediction=train_prediction,
        test_prediction=test_prediction,
        distro_params=result.x,
        rel_train_error=rel_train_error,
        rel_test_error=rel_test_error,
    )

    return result_obj
