"""Compartmental model for ICU length of stay estimation.

This module implements a compartmental model approach where patients flow
through different states (admission -> ICU -> discharge) with specified
transition rates and delays.
"""

import sys
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING or ("coverage" in sys.modules):
    # No JIT during type checking
    def njit(func):
        return func

else:
    from numba import njit


@njit
def calc_its_comp(inc, discharge_rate, transition_rate, delay, init):
    """Calculate ICU occupancy using compartmental model.

    Computes ICU bed occupancy using a compartmental model with
    transition rates, discharge rates, and delays between states.

    Args:
        inc (np.ndarray): Daily incidence/admission data.
        discharge_rate (float): Rate of discharge from ICU.
        transition_rate (float): Rate of transition to ICU.
        delay (float): Delay between admission and ICU entry.
        init (float): Initial ICU occupancy.

    Returns:
        np.ndarray: Predicted ICU occupancy over time.
    """
    int_delay = int(delay)
    beds = inc * transition_rate
    beds = update_beds(beds, init, (1 - discharge_rate))
    intraday_delay = delay - int(delay)

    beds_ext = np.zeros(beds.shape[0] + int_delay + 2, dtype=beds.dtype)
    beds_ext[int_delay + 1 : -1] = beds
    beds_ext[-1] = beds[-1]
    beds = beds_ext

    beds = beds[1:] * (1 - intraday_delay) + beds[:-1] * intraday_delay
    beds = beds[:-1]
    return beds


@njit
def update_beds(beds, init, rate):
    """Update bed occupancy using retention rate.

    Updates the bed occupancy series by applying retention rates
    and initial conditions to model patient flow.

    Args:
        beds (np.ndarray): Bed occupancy array to update.
        init (float): Initial bed count.
        rate (float): Retention rate (1 - discharge_rate).

    Returns:
        np.ndarray: Updated bed occupancy array.
    """
    beds[0] += init
    for i in range(len(beds) - 1):
        beds[i + 1] += beds[i] * rate
    return beds


def mse(pred, real):
    """Calculate mean squared error between predictions and reality.

    Args:
        pred (np.ndarray): Predicted values.
        real (np.ndarray): Real/observed values.

    Returns:
        float: Mean squared error.
    """
    pred = pred[: len(real)]
    return np.mean((pred - real) ** 2)


def objective_function_compartmental(model_config, inc, icu):
    """Objective function for compartmental model optimization.

    Computes the error between predicted and observed ICU occupancy
    for use in optimization algorithms.

    Args:
        model_config (tuple): Model parameters (discharge_rate, transition_rate, delay).
        inc (np.ndarray): Incidence/admission data.
        icu (np.ndarray): Observed ICU occupancy data.

    Returns:
        float: Mean squared error between predicted and observed values.
    """
    discharge_rate, transition_rate, delay = model_config

    pred = calc_its_comp(inc, discharge_rate, transition_rate, delay, init=icu[0])

    return mse(pred[model_config.kernel_width :], icu[model_config.kernel_width :])
