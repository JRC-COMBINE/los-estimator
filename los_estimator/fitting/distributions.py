"""Probability distributions usable as LOS kernels, plus their fit/sampling bounds."""

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

__all__ = [
    "DistributionTypes",
    "Distribution",
    "Distributions",
    "USES_SCALING",
    "SAMPLING_BOUNDS",
    "stretch_bound",
    "sampling_box",
]


class DistributionTypes:
    """String constants for the supported kernel distributions."""

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
    """One fittable kernel distribution.

    uses_scaling: whether the parameter vector carries a trailing
    ``scaling_fac`` that stretches the day axis before evaluating the PDF.
    Only distributions without their own usable scale parameter need it;
    for the others the stretch is degenerate with that scale parameter and
    makes the fit unidentifiable. scaling_init is ignored otherwise.
    """

    name: str
    init_values: list[float]
    boundaries: list[tuple]
    pdf: Callable
    to_string: Callable = lambda x: str(x)
    uses_scaling: bool = False
    scaling_init: float = 1.0


class DistributionsClass:
    """Collection of probability distributions for model fitting."""

    def generate_kernel(self, distro, fun_params, kernel_size):
        """Evaluate `distro`'s PDF over days 0..kernel_size-1 and normalize to sum 1.

        `fun_params` is the distribution parameters, plus a trailing scaling
        factor iff `uses_scaling` is set for this distribution.
        """
        model_config = list(fun_params)
        x = np.arange(kernel_size, dtype=float)
        if self[distro].uses_scaling:
            scaling_fac = model_config.pop()
            x = x * scaling_fac
        pdf = self.get_pdf(distro)
        kernel = pdf(x, *model_config)
        # Normalization is unconditional: the ``1 - cumsum`` survival transform
        # in ``convolutional_model.los_distro_converter`` assumes a normalized kernel.
        result = kernel / kernel.sum()
        return result

    def generate_kernel_batch(self, distro, fun_params, kernel_size):
        """Vectorized `generate_kernel` over `fun_params` of shape (n_samples, ndim)
        (a 1-D vector is treated as one sample). Falls back to a row-wise loop for
        PDFs that don't broadcast over a 2-D day axis (e.g. `sentinel`, which reads
        `len(x)`)."""
        params = np.atleast_2d(np.asarray(fun_params, dtype=float))
        n_samples = params.shape[0]
        x = np.tile(np.arange(kernel_size, dtype=float), (n_samples, 1))
        columns = [params[:, i] for i in range(params.shape[1])]
        if self[distro].uses_scaling:
            scaling_fac = columns.pop()
            x = x * scaling_fac[:, None]
        pdf = self.get_pdf(distro)
        kernel = np.asarray(pdf(x, *[c[:, None] for c in columns]), dtype=float)
        if kernel.shape != x.shape:
            # PDF is not broadcastable over the batch axis (e.g. ``sentinel``).
            return np.array([self.generate_kernel(distro, p, kernel_size) for p in params])
        # Normalization is unconditional, exactly as in ``generate_kernel``.
        return kernel / kernel.sum(axis=1, keepdims=True)

    _distributions = {
        DistributionTypes.LOGNORM: Distribution(
            name=DistributionTypes.LOGNORM,
            init_values=[1, 0],
            boundaries=[(0.05, 2.5), (0.0, 4.0)],
            pdf=lambda x, sigma, μ: lognorm.pdf(x, s=sigma, scale=np.exp(μ)),
            to_string=lambda sigma, μ: f"sigma={sigma}, μ={μ}",
        ),
        DistributionTypes.WEIBULL: Distribution(
            name=DistributionTypes.WEIBULL,
            init_values=[1.5, 15],
            boundaries=[(1.0, 6.0), (0.5, 60.0)],
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
            boundaries=[(1.0, 20.0), (0.2, 30.0)],
            pdf=lambda x, a, s: gamma.pdf(x, a=a, scale=s),
            to_string=lambda a, s: f"a={a}, s={s}",
        ),
        DistributionTypes.BETA: Distribution(
            name=DistributionTypes.BETA,
            init_values=[2, 2],
            boundaries=[(0, None), (0, None)],
            pdf=lambda x, a, b: beta.pdf(x, a=a, b=b),
            to_string=lambda a, b: f"a={a}, b={b}",
            uses_scaling=True,
            scaling_init=0.01,
        ),
        DistributionTypes.CAUCHY: Distribution(
            name=DistributionTypes.CAUCHY,
            init_values=[0.5, 1],
            boundaries=[(0, None), (0, None)],
            pdf=lambda x, μ, s: cauchy.pdf(x, loc=μ, scale=s),
            to_string=lambda μ, s: f"μ={μ}, s={s}",
        ),
        DistributionTypes.T: Distribution(
            name=DistributionTypes.T,
            init_values=[10, 0, 1],
            boundaries=[(1.0, 50.0), (0.0, 60.0), (0.2, 30.0)],
            pdf=lambda x, v, μ, s: t.pdf(x, df=v, loc=μ, scale=s),
            to_string=lambda v, μ, s: f"v={v}, μ={μ}, s={s}",
        ),
        DistributionTypes.INVGAUSS: Distribution(
            name=DistributionTypes.INVGAUSS,
            init_values=[1, 0],
            boundaries=[(0, None), (0, None)],
            pdf=lambda x, μ, loc: invgauss.pdf(x, μ, loc=loc),
            to_string=lambda μ, loc: f"μ={μ}, loc={loc}",
            # Called without ``scale=``, so it is pinned near x ~ 1 and needs
            # the stretch to reach day scales.
            uses_scaling=True,
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
            # Fixed empirical shape: the stretch is its only degree of freedom.
            uses_scaling=True,
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

    def uses_scaling(self, distro_name):
        """Whether the distribution's parameter vector carries a scaling factor."""
        return self[distro_name].uses_scaling

    def n_parameters(self, distro_name, with_scaling=True):
        """Number of fitted parameters, optionally including the scaling factor."""
        distribution = self[distro_name]
        n = len(distribution.boundaries)
        if with_scaling and distribution.uses_scaling:
            n += 1
        return n

    def to_string(self, distro_name, params):
        """Returns the to_string function for the given distribution type."""
        params = [float(p) for p in params]
        params = [float(f"{p:.2g}") for p in params]
        res = ""
        if self[distro_name].uses_scaling and len(params) > 0:
            res += f", x_scale={params[-1]}"
            params = params[:-1]

        return f"{self[distro_name].to_string(*params)}" + res


Distributions = DistributionsClass()

USES_SCALING = frozenset(name for name, d in DistributionsClass._distributions.items() if d.uses_scaling)


#: Finite sampling boxes for the distribution parameters, used **only** as domain
#: guards when rejection-sampling the Laplace posterior. They are deliberately
#: separate from :attr:`Distribution.boundaries`, which drives the optimizer and
#: contains open bounds such as ``(0, None)`` that cannot be sampled from. They
#: are also deliberately generous: draws come from ``N(best_params, cov)``
#: concentrated at the optimum, and tightening a box would truncate genuine
#: posterior mass and narrow the bands.
SAMPLING_BOUNDS: dict[str, list[tuple[float, float]]] = {
    "lognorm": [(0.05, 2.5), (0.0, 4.0)],  # (sigma, mu); median exp(mu) = 1..55 d
    "weibull": [(1.0, 6.0), (0.5, 60.0)],  # (k, lambda)
    "gaussian": [(0.0, 60.0), (0.5, 30.0)],  # (mu, sigma)
    "exponential": [(0.005, 2.0)],  # rate -> mean 0.5..200 d
    "gamma": [(1.0, 20.0), (0.2, 30.0)],  # (a, s); a >= 1 avoids the pole at x=0
    "beta": [(1.0, 8.0), (1.0, 8.0)],  # (a, b); a,b >= 1 avoids the poles
    "cauchy": [(0.0, 60.0), (0.2, 30.0)],  # (loc, scale)
    "t": [(1.0, 50.0), (0.0, 60.0), (0.2, 30.0)],  # (df, loc, scale)
    "invgauss": [(0.05, 20.0), (-5.0, 0.0)],  # (mu, loc); loc > 0 can empty the kernel
    "linear": [(1.0, 200.0)],  # L
    "sentinel": [],
}


def stretch_bound(distro: str, kernel_width: int) -> tuple[float, float]:
    """Sampling (low, high) box for the trailing scaling_fac; only meaningful
    for distributions in USES_SCALING."""
    if distro == "beta":
        # x * s must stay inside beta's [0, 1] support
        return (1.0 / (4 * kernel_width), 1.0 / (kernel_width - 1))
    return {"invgauss": (0.01, 2.0), "sentinel": (0.1, 5.0)}[distro]


def sampling_box(distro: str, kernel_width: int) -> list[tuple[float, float]]:
    """SAMPLING_BOUNDS plus the scaling-factor bound (if any), one (low, high)
    pair per fitted parameter, element-wise aligned with the parameter vector."""
    if distro not in SAMPLING_BOUNDS:
        raise ValueError(f"No sampling bounds defined for distribution: {distro}")
    box = list(SAMPLING_BOUNDS[distro])
    if Distributions[distro].uses_scaling:
        box.append(stretch_bound(distro, kernel_width))
    return box
