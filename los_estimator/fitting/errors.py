"""Error metrics used for model fitting and evaluation."""

from numpy.typing import NDArray
import sys
from typing import TYPE_CHECKING

import numpy as np

from numba import njit

__all__ = [
    "ErrorType",
    "ErrorFunctions",
]


class ErrorType:
    """String constants for the supported error metrics."""

    MSE = "mse"
    WEIGHTED_MSE = "weighted_mse"
    MAE = "mae"
    RMSE = "rmse"
    MAPE = "mape"
    SMAPE = "smape"
    R2 = "r2"
    INC_ERROR = "inc_error"


class _ErrorFunctions:
    """Registry of pluggable error/loss functions, selected via `model_config.error_fun`."""

    def cap_err(y_true: NDArray, y_pred: NDArray, cap, a=0.02):
        # NOTE: `cap`/`a` are unused after the last two `weights =` reassignments
        # overwrite them - effectively just a normalized-incidence weighting,
        # same as inc_error below. Left as-is (not called via `errors` dict).
        weights = np.exp(((y_true - cap) / cap) * a)
        weights = np.abs((y_true - cap) / cap)
        weights = y_true.copy()
        weights /= weights.sum()
        return weights * np.abs(y_true - y_pred)

    @njit
    def inc_error(y_true, y_pred):
        """Absolute error weighted by each point's share of total incidence."""
        weights = y_true / y_true.sum()
        return weights * np.abs(y_true - y_pred)

    @njit
    def weighted_mse(x, y):
        """MSE with exponentially increasing weight toward the end of the series."""
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
