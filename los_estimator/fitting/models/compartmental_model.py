"""Compartmental LOS model: patients flow admission -> ICU -> discharge with
transition/discharge rates and a delay."""

import sys
from typing import TYPE_CHECKING

import numpy as np

from numba import njit


@njit
def calc_its_comp(inc, discharge_rate, transition_rate, delay, init):
    """Predict ICU occupancy from admissions via the compartmental model.
    `delay` (days) may be fractional; the integer and intraday parts are
    applied separately (shift + linear interpolation)."""
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
    """Roll `beds` forward in place: each day retains `rate` (= 1 - discharge_rate)
    of the previous day's occupancy, seeded by `init`."""
    beds[0] += init
    for i in range(len(beds) - 1):
        beds[i + 1] += beds[i] * rate
    return beds


def mse(pred, real):
    pred = pred[: len(real)]
    return np.mean((pred - real) ** 2)


def objective_function_compartmental(model_config, inc, icu):
    """Optimizer objective: MSE of compartmental-model prediction vs observed ICU occupancy."""
    discharge_rate, transition_rate, delay = model_config

    pred = calc_its_comp(inc, discharge_rate, transition_rate, delay, init=icu[0])

    return mse(pred[model_config.kernel_width :], icu[model_config.kernel_width :])
