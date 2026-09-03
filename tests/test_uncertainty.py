"""Tests for the Laplace-approximation uncertainty quantification.

The equivalence tests are the load-bearing ones: a batch kernel generator or a
batch convolution that is only *nearly* equivalent to the scalar path produces
plausible-looking but wrong bands that nothing else would catch.
"""

import numpy as np
import pytest

from los_estimator.fitting.distributions import Distributions, sampling_box
from los_estimator.fitting.los_fitter import (
    combine_past_kernel,
    fit_convolution,
    finite_difference_hessian,
    invert_loss_hessian,
)
from los_estimator.fitting.models.convolutional_model import calc_its_convolution
from los_estimator.fitting.uncertainty import (
    UncertaintyParams,
    compute_window_uncertainty,
    convolve_batch,
)

KERNEL_WIDTH = 60
TOL = 1e-13

ALL_DISTROS = [
    "lognorm",
    "weibull",
    "gaussian",
    "exponential",
    "gamma",
    "beta",
    "cauchy",
    "t",
    "invgauss",
    "linear",
    "sentinel",
]


def draw_params(distro, n_samples, kernel_width=KERNEL_WIDTH, seed=0):
    """Uniform parameter draws inside the distribution's sampling box."""
    box = sampling_box(distro, kernel_width)
    rng = np.random.default_rng(seed)
    lows = np.array([b[0] for b in box])
    highs = np.array([b[1] for b in box])
    return rng.uniform(lows, highs, size=(n_samples, len(box)))


def synthetic_series(n_days=200, seed=3):
    """Admissions plus the occupancy they generate through a known kernel."""
    rng = np.random.default_rng(seed)
    admissions = 20 + 10 * np.sin(np.arange(n_days) / 15.0) + rng.normal(0, 1.5, n_days)
    admissions = np.clip(admissions, 1.0, None)
    kernel = Distributions.generate_kernel("gaussian", [8.0, 4.0], KERNEL_WIDTH)
    occupancy = calc_its_convolution(admissions, kernel)
    occupancy = occupancy + rng.normal(0, 2.0, n_days)
    return admissions, occupancy


@pytest.mark.parametrize("distro", ALL_DISTROS)
def test_generate_kernel_batch_matches_scalar(distro):
    """The batch kernel generator must reproduce the scalar one exactly."""
    params = draw_params(distro, 25, seed=11)
    batch = Distributions.generate_kernel_batch(distro, params, KERNEL_WIDTH)
    scalar = np.array([Distributions.generate_kernel(distro, p, KERNEL_WIDTH) for p in params])
    assert batch.shape == scalar.shape
    assert np.all(np.isfinite(batch)), f"{distro}: non-finite batch kernel"
    delta = np.max(np.abs(batch - scalar))
    assert delta < TOL, f"{distro}: max |batch - scalar| = {delta:.3e}"


@pytest.mark.parametrize("distro", ["lognorm", "gaussian", "gamma", "linear", "t", "invgauss"])
def test_convolve_batch_matches_scalar(distro):
    """The batch convolution must reproduce `calc_its_convolution` exactly."""
    admissions, _ = synthetic_series()
    params = draw_params(distro, 10, seed=5)
    kernels = Distributions.generate_kernel_batch(distro, params, KERNEL_WIDTH)

    batch = convolve_batch(admissions, kernels)
    scalar = np.array([calc_its_convolution(admissions, k) for k in kernels])
    delta = np.max(np.abs(batch - scalar))
    scale = np.max(np.abs(scalar))
    assert delta < TOL * max(scale, 1.0), f"{distro}: max |batch - scalar| = {delta:.3e} (scale {scale:.3e})"


@pytest.mark.parametrize("distro", ["lognorm", "gaussian", "gamma", "linear"])
def test_convolve_batch_matches_scalar_with_past_kernels(distro):
    """Same, for the `iterative_kernel_fit` (stacked past kernels) path."""
    admissions, _ = synthetic_series()
    past_params = draw_params(distro, KERNEL_WIDTH, seed=7)
    past_kernels = Distributions.generate_kernel_batch(distro, past_params, KERNEL_WIDTH)
    params = draw_params(distro, 10, seed=8)
    kernels = Distributions.generate_kernel_batch(distro, params, KERNEL_WIDTH)

    batch = convolve_batch(admissions, kernels, past_kernels=past_kernels)
    scalar = np.array([calc_its_convolution(admissions, combine_past_kernel(past_kernels, k)) for k in kernels])
    delta = np.max(np.abs(batch - scalar))
    scale = np.max(np.abs(scalar))
    assert delta < TOL * max(scale, 1.0), f"{distro}: max |batch - scalar| = {delta:.3e} (scale {scale:.3e})"


def _make_fit(distro="gaussian", past_kernels=None):
    """Fit one window of the synthetic series."""
    admissions, occupancy = synthetic_series()
    train = (admissions[:140], occupancy[:140])
    test = (admissions[100:180], occupancy[100:180])
    result = fit_convolution(
        distro,
        train,
        test,
        KERNEL_WIDTH,
        past_kernels=past_kernels,
    )
    return result


def test_covariance_is_estimated_and_scaled():
    """A well-conditioned fit yields a finite, positive-definite covariance."""
    result = _make_fit()
    assert result.covariance is not None, result.covariance_source
    assert np.all(np.isfinite(result.covariance))
    np.linalg.cholesky(result.covariance)  # raises if not positive definite
    assert result.n_residuals == 140 - KERNEL_WIDTH


def test_non_psd_covariance_is_handled_without_raising():
    """A non-PSD / singular covariance must be refused, not sampled from."""
    result = _make_fit()
    ndim = len(result.distro_params)
    params = UncertaintyParams(enabled=True, n_samples=50, seed=0)
    rng = np.random.default_rng(0)

    result.covariance = -np.eye(ndim)  # negative definite
    assert compute_window_uncertainty(result, KERNEL_WIDTH, params, rng) is False
    assert result.train_lower is None

    result.covariance = np.zeros((ndim, ndim))  # singular
    assert compute_window_uncertainty(result, KERNEL_WIDTH, params, rng) is False

    result.covariance = None  # missing
    assert compute_window_uncertainty(result, KERNEL_WIDTH, params, rng) is False
    assert result.kernel_lower is None


def test_zero_hessian_is_refused():
    """A numerically zero Hessian must not produce an infinite band."""
    hessian = finite_difference_hessian(lambda p: 1.0, np.array([1.0]))
    inverse, reason = invert_loss_hessian(hessian)
    assert inverse is None
    assert "zero" in reason


@pytest.mark.parametrize("with_past", [False, True])
def test_bands_bracket_prediction(with_past):
    """`lower <= prediction <= upper` pointwise, and every band is finite."""
    past_kernels = None
    if with_past:
        past_params = draw_params("gaussian", KERNEL_WIDTH, seed=2)
        past_kernels = Distributions.generate_kernel_batch("gaussian", past_params, KERNEL_WIDTH)
    result = _make_fit(past_kernels=past_kernels)
    params = UncertaintyParams(enabled=True, n_samples=500, confidence_interval=(5, 95), seed=42)
    ok = compute_window_uncertainty(
        result, KERNEL_WIDTH, params, np.random.default_rng(42), past_kernels=past_kernels
    )
    assert ok, result.covariance_source

    for lower, upper, prediction in [
        (result.train_lower, result.train_upper, result.train_prediction),
        (result.test_lower, result.test_upper, result.test_prediction),
        (result.kernel_lower, result.kernel_upper, result.kernel),
    ]:
        assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
        assert lower.shape == prediction.shape
        assert np.all(lower <= upper)
        assert np.all(lower <= prediction + 1e-9), f"lower above prediction by {np.max(lower - prediction):.3e}"
        assert np.all(upper >= prediction - 1e-9), f"upper below prediction by {np.max(prediction - upper):.3e}"

    assert 0.0 < result.acceptance_rate <= 1.0
    assert result.train_sigma > 0 and result.test_sigma > 0
    # decision 4: the test band is calibrated with the out-of-sample residual
    assert result.test_sigma != result.train_sigma
