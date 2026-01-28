"""Error functions for model fitting and evaluation.

This module provides various error metrics that can be used to evaluate
the quality of length of stay model fits, including standard statistical
measures and domain-specific error functions.
"""

from numpy.typing import NDArray
import sys
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING or ("coverage" in sys.modules):
    # No JIT during type checking
    def njit(func):
        return func

else:
    from numba import njit

__all__ = [
    "ErrorType",
    "ErrorFunctions",
]


class ErrorType:
    """Enum for available error function types.

    Defines constants for all supported error metrics that can be used
    for model fitting and evaluation.
    """

    MSE = "mse"
    WEIGHTED_MSE = "weighted_mse"
    MAE = "mae"
    RMSE = "rmse"
    MAPE = "mape"
    SMAPE = "smape"
    R2 = "r2"
    INC_ERROR = "inc_error"


class _ErrorFunctions:
    """Collection of error functions for model fitting.

    Provides various error metrics and loss functions that can be used
    to evaluate model performance and guide optimization.
    """

    def cap_err(y_true: NDArray, y_pred: NDArray, cap, a=0.02):
        """Capacity-weighted error function.

        Args:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.
            cap (float): Capacity threshold for weighting.
            a (float, optional): Weighting parameter. Defaults to 0.02.

        Returns:
            np.ndarray: Weighted absolute errors.
        """
        weights = np.exp(((y_true - cap) / cap) * a)
        weights = np.abs((y_true - cap) / cap)
        weights = y_true.copy()
        weights /= weights.sum()
        return weights * np.abs(y_true - y_pred)

    @njit
    def inc_error(y_true, y_pred):
        """Incidence-weighted error function.

        Computes weighted absolute error where weights are proportional
        to the true values, emphasizing periods with higher incidence.

        Args:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            np.ndarray: Weighted absolute errors.
        """
        weights = y_true / y_true.sum()
        return weights * np.abs(y_true - y_pred)

    @njit
    def weighted_mse(x, y):
        """Weighted mean squared error with time-based weighting.

        Applies higher weights to more recent observations in the time series.

        Args:
            x (np.ndarray): True values.
            y (np.ndarray): Predicted values.

        Returns:
            float: Weighted mean squared error.
        """
        le = len(x)
        weights = np.exp(np.linspace(0, 2, le))
        weights /= weights.sum()
        return np.sum(((x - y) ** 2) * weights)

    @njit
    def mse(x, y):
        return np.mean((x - y) ** 2)

    @njit
    def mae(x, y):
        return np.mean(np.abs(x - y))

    @njit
    def rmse(x, y):
        return np.sqrt(np.mean((x - y) ** 2))

    @njit
    def mape(x, y):
        return np.mean(np.abs((x - y) / x))

    @njit
    def smape(x, y):
        return np.mean(np.abs(x - y) / (np.abs(x) + np.abs(y)) * 2)

    @njit
    def r2(x, y):
        ss_res = np.sum((x - y) ** 2)
        ss_tot = np.sum((x - np.mean(x)) ** 2)
        return 1 - (ss_res / ss_tot)

    errors = {
        ErrorType.MSE: mse,
        ErrorType.WEIGHTED_MSE: weighted_mse,
        ErrorType.MAE: mae,
        ErrorType.RMSE: rmse,
        ErrorType.MAPE: mape,
        ErrorType.SMAPE: smape,
        ErrorType.R2: r2,
    }

    def __getitem__(self, error_fun):
        """Returns the appropriate error function based on the input string."""
        if error_fun in self.errors:
            return self.errors[error_fun]
        else:
            raise ValueError(f"Unknown Error Function: {error_fun}")


ErrorFunctions = _ErrorFunctions()
