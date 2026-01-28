"""Distribution types and probability density functions for LOS modeling.

This module provides probability distributions that can be used as
kernels for length of stay estimation, including standard statistical
distributions and custom distributions specific to hospital data.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import (
    beta,
    cauchy,
    expon,
    gamma,
    invgauss,
    lognorm,
    norm,
    t,
    weibull_min,
)
from .sentinel_distro import sentinel_los_berlin

__all__ = ["DistributionTypes", "Distribution", "Distributions"]


class DistributionTypes:
    """Enum for available distribution types.

    Defines constants for all supported distribution types that can be
    used as length of stay kernels in the modeling process.
    """

    LOGNORM = "lognorm"
    WEIBULL = "weibull"
    GAUSSIAN = "gaussian"
    EXPONENTIAL = "exponential"
    GAMMA = "gamma"
    BETA = "beta"
    CAUCHY = "cauchy"
    T = "t"
    INVGAUSS = "invgauss"
    LINEAR = "linear"
    SENTINEL = "sentinel"


@dataclass
class Distribution:
    """Data class for distribution function information.

    Contains all necessary information to define and use a probability
    distribution for length of stay modeling.

    Attributes:
        name (str): Name identifier for the distribution.
        init_values (list): Initial parameter values for optimization.
        boundaries (list): Parameter bounds for constrained optimization.
        pdf (Callable): Probability density function.
    """

    name: str
    init_values: list[float]
    boundaries: list[tuple]
    pdf: Callable
    to_string: Callable = lambda x: str(x)


class DistributionsClass:
    """Collection of probability distributions for model fitting."""

    def generate_kernel(self, distro, fun_params, kernel_size):
        """Generate a kernel using the specified distribution."""
        *model_config, scaling_fac = fun_params
        pdf = self.get_pdf(distro)
        x = np.arange(kernel_size, dtype=float) * scaling_fac
        kernel = pdf(x, *model_config)
        result = kernel / kernel.sum()
        return result

    _distributions = {
        DistributionTypes.LOGNORM: Distribution(
            name=DistributionTypes.LOGNORM,
            init_values=[1, 0],
            boundaries=[(0, None), (0, None)],
            pdf=lambda x, sigma, μ: lognorm.pdf(x, s=sigma, scale=np.exp(μ)),
            to_string=lambda sigma, μ: f"sigma={sigma}, μ={μ}",
        ),
        DistributionTypes.WEIBULL: Distribution(
            name=DistributionTypes.WEIBULL,
            init_values=[1, 15],
            boundaries=[(1, None), (0, None)],
            pdf=lambda x, k, λ: weibull_min.pdf(x, c=k, scale=λ),
            to_string=lambda k, λ: f"k={k}, λ={λ}",
        ),
        DistributionTypes.GAUSSIAN: Distribution(
            name=DistributionTypes.GAUSSIAN,
            init_values=[0, 1],
            boundaries=[(0, None), (0, None)],
            pdf=lambda x, μ, sigma: norm.pdf(x, loc=μ, scale=sigma),
            to_string=lambda μ, sigma: f"μ={μ}, σ={sigma}",
        ),
        DistributionTypes.EXPONENTIAL: Distribution(
            name=DistributionTypes.EXPONENTIAL,
            init_values=[1],
            boundaries=[(0.001, None)],
            pdf=lambda x, λ: expon.pdf(x, scale=1 / λ),
            to_string=lambda λ: f"λ={λ}",
        ),
        DistributionTypes.GAMMA: Distribution(
            name=DistributionTypes.GAMMA,
            init_values=[2, 2],
            boundaries=[(0, None), (0, None)],
            pdf=lambda x, a, s: gamma.pdf(x, a=a, scale=s),
            to_string=lambda a, s: f"a={a}, s={s}",
        ),
        DistributionTypes.BETA: Distribution(
            name=DistributionTypes.BETA,
            init_values=[2, 2],
            boundaries=[(0, None), (0, None)],
            pdf=lambda x, a, b: beta.pdf(x, a=a, b=b),
            to_string=lambda a, b: f"a={a}, b={b}",
        ),
        DistributionTypes.CAUCHY: Distribution(
            name=DistributionTypes.CAUCHY,
            init_values=[0, 1],
            boundaries=[(0, None), (0, None)],
            pdf=lambda x, μ, s: cauchy.pdf(x, loc=μ, scale=s),
            to_string=lambda μ, s: f"μ={μ}, s={s}",
        ),
        DistributionTypes.T: Distribution(
            name=DistributionTypes.T,
            init_values=[10, 0, 1],
            boundaries=[(0, None), (0, None), (0, None)],
            pdf=lambda x, v, μ, s: t.pdf(x, df=v, loc=μ, scale=s),
            to_string=lambda v, μ, s: f"v={v}, μ={μ}, s={s}",
        ),
        DistributionTypes.INVGAUSS: Distribution(
            name=DistributionTypes.INVGAUSS,
            init_values=[1, 0],
            boundaries=[(0, None), (0, None)],
            pdf=lambda x, μ, loc: invgauss.pdf(x, μ, loc=loc),
            to_string=lambda μ, loc: f"μ={μ}, loc={loc}",
        ),
        DistributionTypes.LINEAR: Distribution(
            name=DistributionTypes.LINEAR,
            init_values=[40],
            boundaries=[(0, None)],
            pdf=lambda x, L: np.clip(-x / L + 1, 0, None),
            to_string=lambda L: f"L={L}",
        ),
        DistributionTypes.SENTINEL: Distribution(
            name=DistributionTypes.SENTINEL,
            init_values=[],
            boundaries=[],
            pdf=lambda x: np.asarray(sentinel_los_berlin[: len(x)], dtype=float),
            to_string=lambda: "Sentinel Distribution",
        ),
    }

    def __getitem__(self, distro_name):
        """Get the distribution by name."""
        if distro_name in self._distributions:
            return self._distributions[distro_name]
        else:
            raise ValueError(f"Unknown Distribution: {distro_name}")

    def get_pdf(self, distro_name):
        """Returns the PDF function for the given distribution type."""
        return self[distro_name].pdf

    def to_string(self, distro_name, params):
        """Returns the to_string function for the given distribution type."""
        params = [float(p) for p in params]
        params = [float(f"{p:.2g}") for p in params]
        res = ""
        if len(params) > 0:
            res += f", x_scale={params[-1]}"

        return f"{self[distro_name].to_string(*params[:-1])}" + res


Distributions = DistributionsClass()
