"""Laplace-approximation UQ for convolution fits: sample parameters from the
posterior covariance (from `los_fitter`), propagate through kernel + convolution,
reduce to percentile bands on the kernel and train/test predictions.

Fully vectorized/batched: the scalar path costs ~2.2ms/sample and would put a
full run in the hour range.

Limitation: observation noise is added as iid Gaussian, but occupancy residuals
are strongly autocorrelated, so this understates band width (known cause of
under-coverage).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .distributions import Distributions, sampling_box
from .models.convolutional_model import (
    convolve_2d_changing_kernel,
    los_distro_converter,
)

logger = logging.getLogger("los_estimator")

__all__ = [
    "UncertaintyParams",
    "build_A",
    "convolve_batch",
    "sample_parameters",
    "compute_window_uncertainty",
]

#: Distributions that have their own fitter and parameter meaning; UQ is skipped.
UNSUPPORTED_DISTROS = frozenset({"compartmental"})

#: Reasons already logged, so a skip is reported once per distribution/reason.
_reported: set = set()


@dataclass
class UncertaintyParams:
    """Settings for the uncertainty pass; carries its own defaults so fitting
    code can be driven directly without an `UncertaintyConfig`."""

    enabled: bool = False
    n_samples: int = 1000
    confidence_interval: Tuple[float, float] = (5.0, 95.0)
    distributions: Optional[Sequence[str]] = None
    seed: Optional[int] = None
    max_batches: int = 10

    @classmethod
    def from_config(cls, config: Optional[object]) -> "UncertaintyParams":
        """Build from anything duck-typed like `UncertaintyConfig` (avoids a
        dependency on `los_estimator.config`); disabled defaults if config is None."""
        if config is None:
            return cls()
        distributions = config.distributions
        return cls(
            enabled=config.enabled,
            n_samples=config.n_samples,
            confidence_interval=tuple(config.confidence_interval),
            distributions=list(distributions) if distributions else None,
            seed=config.seed,
        )

    def covers(self, distro: str) -> bool:
        """Whether UQ should run for `distro` (enabled, supported, and in scope)."""
        if not self.enabled or distro in UNSUPPORTED_DISTROS:
            return False
        if self.distributions is not None and distro not in self.distributions:
            return False
        return True


def _report_once(distro: str, reason: str) -> None:
    """Log a skip reason once per distribution."""
    key = (distro, reason)
    if key in _reported:
        logger.debug(f"uncertainty skipped for {distro}: {reason}")
        return
    _reported.add(key)
    logger.info(f"uncertainty skipped for {distro}: {reason}")


def build_A(admissions: np.ndarray, kernel_width: int) -> np.ndarray:
    """Sliding-admissions matrix for a fixed-kernel convolution: A[i, p] =
    admissions[i - p] (0 for i < p), so A @ kernel == calc_its_convolution."""
    admissions = np.asarray(admissions, dtype=float)
    n = admissions.shape[0]
    A = np.zeros((n, kernel_width))
    for p in range(kernel_width):
        A[p:, p] = admissions[: n - p]
    return A


def convolve_batch(
    admissions: np.ndarray,
    kernels: np.ndarray,
    past_kernels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Batched `calc_its_convolution`: one matrix product reproduces it exactly
    (survival transform + zeroing the first kernel_width points) for every row
    of `kernels`, shape (n_samples, kernel_width) -> (n_samples, n).

    With `past_kernels` (iterative_kernel_fit), t < n_past is governed by the
    past rows (identical across samples) and t >= n_past by the new kernel;
    computed separately and summed."""
    admissions = np.asarray(admissions, dtype=float)
    kernels = np.atleast_2d(np.asarray(kernels, dtype=float))
    kernel_width = kernels.shape[1]
    survival = los_distro_converter(kernels)

    if past_kernels is None:
        predictions = (build_A(admissions, kernel_width) @ survival.T).T
    else:
        past_kernels = np.atleast_2d(np.asarray(past_kernels, dtype=float))
        n_past = past_kernels.shape[0]
        # Contribution of admissions at t >= n_past, governed by the new kernel.
        admissions_new = admissions.copy()
        admissions_new[:n_past] = 0.0
        predictions = (build_A(admissions_new, kernel_width) @ survival.T).T
        # Contribution of admissions at t < n_past, identical for every sample.
        admissions_past = admissions.copy()
        admissions_past[n_past:] = 0.0
        past_stack = np.vstack([past_kernels, kernels[0]])
        past_prediction = convolve_2d_changing_kernel(
            admissions_past, los_distro_converter(past_stack)
        )
        predictions = predictions + past_prediction[None, :]

    # match calc_its_convolution's transient handling
    predictions[:, :kernel_width] = 0.0
    return np.asarray(predictions)


def sample_parameters(
    mean: np.ndarray,
    covariance: np.ndarray,
    box: Sequence[Tuple[float, float]],
    n_samples: int,
    rng: np.random.Generator,
    max_batches: int = 10,
) -> Tuple[Optional[np.ndarray], float]:
    """Rejection-sample from N(mean, covariance) inside `box` (a domain guard:
    draws outside it give empty/NaN/infinite kernels). Batches until n_samples
    accepted or max_batches tried. Returns (draws or None, acceptance_rate)."""
    mean = np.asarray(mean, dtype=float)
    lows = np.array([b[0] for b in box], dtype=float)
    highs = np.array([b[1] for b in box], dtype=float)

    accepted = []
    n_drawn = 0
    n_accepted = 0
    for _ in range(max_batches):
        draws = rng.multivariate_normal(
            mean, covariance, size=n_samples, method="cholesky"
        )
        n_drawn += n_samples
        inside = np.all((draws >= lows) & (draws <= highs), axis=1)
        keep = draws[inside]
        if keep.size:
            accepted.append(keep)
            n_accepted += keep.shape[0]
        if n_accepted >= n_samples:
            break
    rate = n_accepted / n_drawn if n_drawn else 0.0
    if not accepted:
        return None, rate
    return np.vstack(accepted)[:n_samples], rate


def _percentiles(
    ensemble: np.ndarray, confidence_interval
) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce an ensemble to lower/upper percentile bands along axis 0."""
    low, high = confidence_interval
    return (
        np.percentile(ensemble, low, axis=0),
        np.percentile(ensemble, high, axis=0),
    )


def _noise_sigma(
    observed: np.ndarray, predicted: np.ndarray, kernel_width: int
) -> float:
    """RMSE over the evaluated slice (past kernel_width runup), recomputed
    explicitly since the stored fit error may come from a non-MSE error_fun."""
    observed = np.asarray(observed, dtype=float)[kernel_width:]
    predicted = np.asarray(predicted, dtype=float)[kernel_width:]
    if observed.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def compute_window_uncertainty(
    fit_result,
    kernel_width: int,
    params: UncertaintyParams,
    rng: np.random.Generator,
    past_kernels: Optional[np.ndarray] = None,
) -> bool:
    """Attach percentile bands to `fit_result` in place; returns True if attached.
    Skips cleanly (band fields left None, reason logged once) when the covariance
    is missing/unusable, no sampling box exists, or fewer than two draws survive
    rejection sampling."""
    distro = fit_result.distro
    if fit_result.covariance is None:
        _report_once(distro, f"no covariance ({fit_result.covariance_source})")
        return False

    covariance = np.atleast_2d(np.asarray(fit_result.covariance, dtype=float))
    mean = np.asarray(fit_result.distro_params, dtype=float)
    if covariance.shape != (mean.size, mean.size):
        _report_once(distro, "covariance shape does not match the parameter vector")
        return False
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        # multivariate_normal accepts a non-PSD matrix silently; refuse it here.
        _report_once(distro, "covariance is not positive definite")
        return False

    try:
        box = sampling_box(distro, kernel_width)
    except ValueError:
        _report_once(distro, "no sampling bounds defined")
        return False
    if len(box) != mean.size:
        _report_once(
            distro, f"sampling box has {len(box)} entries for {mean.size} parameters"
        )
        return False

    samples, acceptance_rate = sample_parameters(
        mean, covariance, box, params.n_samples, rng, max_batches=params.max_batches
    )
    fit_result.acceptance_rate = acceptance_rate
    if samples is None or samples.shape[0] < 2:
        _report_once(
            distro,
            f"fewer than 2 draws survived rejection (rate={acceptance_rate:.3g})",
        )
        return False

    kernels = Distributions.generate_kernel_batch(distro, samples, kernel_width)
    finite = np.all(np.isfinite(kernels), axis=1)
    if finite.sum() < 2:
        _report_once(distro, "fewer than 2 finite kernels")
        return False
    kernels = kernels[finite]
    samples = samples[finite]

    x_train, y_train = fit_result.train_data
    x_test, y_test = fit_result.test_data

    train_ensemble = convolve_batch(x_train, kernels, past_kernels)
    test_ensemble = convolve_batch(x_test, kernels, past_kernels)

    train_sigma = _noise_sigma(y_train, fit_result.train_prediction, kernel_width)
    test_sigma = _noise_sigma(y_test, fit_result.test_prediction, kernel_width)

    # iid Gaussian observation noise. Occupancy residuals are strongly
    # autocorrelated, so this understates the true band width -- a known
    # limitation and a likely contributor to under-coverage.
    if np.isfinite(train_sigma) and train_sigma > 0:
        train_ensemble = train_ensemble + rng.normal(
            0.0, train_sigma, train_ensemble.shape
        )
    if np.isfinite(test_sigma) and test_sigma > 0:
        test_ensemble = test_ensemble + rng.normal(0.0, test_sigma, test_ensemble.shape)

    fit_result.parameter_samples = samples
    fit_result.train_sigma = train_sigma
    fit_result.test_sigma = test_sigma
    fit_result.kernel_lower, fit_result.kernel_upper = _percentiles(
        kernels, params.confidence_interval
    )
    fit_result.train_lower, fit_result.train_upper = _percentiles(
        train_ensemble, params.confidence_interval
    )
    fit_result.test_lower, fit_result.test_upper = _percentiles(
        test_ensemble, params.confidence_interval
    )
    fit_result.uq_past_kernels = (
        None if past_kernels is None else np.array(past_kernels, dtype=float)
    )
    return True
