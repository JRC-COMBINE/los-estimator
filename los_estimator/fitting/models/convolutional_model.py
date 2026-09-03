"""Predict ICU occupancy by convolving admissions with a (possibly time-varying) LOS kernel."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from numba import njit


def los_distro_converter(los):
    """Discharge-time distribution (los, 2D) -> presence-in-ICU survival
    function (1 - cumsum), monotonically decreasing. Assumes los sums to 1."""
    if len(los.shape) == 1:
        raise Exception("los_distro must be 2D")
    los2 = 1 - np.cumsum(los, axis=1)
    return los2


def calc_its_convolution(admissions, los_distro1):
    """Predict ICU occupancy from admissions convolved with (possibly per-window)
    LOS distribution(s). A 1D los_distro1 is broadcast to all timesteps."""
    if len(los_distro1.shape) == 1:
        los_distro1 = los_distro1[None, :]
    los_distro = los_distro_converter(los_distro1)
    its = convolve_2d_changing_kernel(admissions, los_distro)
    its[: los_distro.shape[1]] = 0  # Remove initial transient response
    return its


@njit
def convolve_2d_changing_kernel(admissions, los_distro):
    """Convolve admissions with a kernel that can change per timestep.
    los_distro has one row per timestep (or fewer, clamped via i_kernel)."""
    adm_len = admissions.shape[0]
    n_kernel, kernel_len = los_distro.shape

    result = np.zeros(adm_len)
    for t in range(adm_len):
        for kernel_pos in range(min(kernel_len, adm_len - t)):
            i_kernel = min(t, n_kernel - 1)
            result[t + kernel_pos] += admissions[t] * los_distro[i_kernel, kernel_pos]
    return result
