"""Convolutional model for length of stay estimation.

This module implements convolution-based models that use admission data
and length of stay distributions to predict ICU occupancy through
mathematical convolution operations.
"""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    # No JIT during type checking or when running coverage
    def njit(func):
        return func

else:
    from numba import njit


def njit(func):
    return func


def los_distro_converter(los):
    """Convert discharge distribution to ICU presence distribution.

    Converts from distribution of discharge to distribution of presence in ICU
    (monotonically decreasing) by computing the cumulative survival function.

    Args:
        los (np.ndarray): Length of stay distribution (2D array).

    Returns:
        np.ndarray: ICU presence distribution.

    Raises:
        Exception: If los is not 2D.
    """
    if len(los.shape) == 1:
        raise Exception("los_distro must be 2D")
    los2 = 1 - np.cumsum(los, axis=1)
    return los2


def calc_its_convolution(admissions, los_distro1):
    """Calculate ICU occupancy using convolution with LOS distribution.

    Computes the ICU occupancy time series by convolving admission data
    with length of stay distributions, accounting for variable kernels over time.

    Args:
        admissions (np.ndarray): Daily admission counts.
        los_distro1 (np.ndarray): Length of stay distribution(s).

    Returns:
        np.ndarray: Predicted ICU occupancy time series.
    """
    if len(los_distro1.shape) == 1:
        los_distro1 = los_distro1[None, :]
    los_distro = los_distro_converter(los_distro1)
    its = convolve_2d_changing_kernel(admissions, los_distro)
    its[: los_distro.shape[1]] = 0  # Remove initial transient response
    return its


@njit
def convolve_2d_changing_kernel(admissions, los_distro):
    """Perform convolution with time-varying kernels.

    Efficiently computes convolution where the kernel can change over time,
    using Numba JIT compilation for performance.

    Args:
        admissions (np.ndarray): Admission time series.
        los_distro (np.ndarray): Time-varying LOS distributions (2D).

    Returns:
        np.ndarray: Convolved result representing ICU occupancy.
    """
    adm_len = admissions.shape[0]
    n_kernel, kernel_len = los_distro.shape

    result = np.zeros(adm_len)
    for t in range(adm_len):
        for kernel_pos in range(min(kernel_len, adm_len - t)):
            i_kernel = min(t, n_kernel - 1)
            result[t + kernel_pos] += admissions[t] * los_distro[i_kernel, kernel_pos]
    return result
