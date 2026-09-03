"""Per-window fitting: convolution and compartmental model optimization, plus
Laplace-approximation covariance estimation for uncertainty quantification."""

from __future__ import annotations

import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from typing import Callable, Optional, Tuple, List

from los_estimator.fitting.models.compartmental_model import calc_its_comp
from los_estimator.fitting.models.convolutional_model import calc_its_convolution

from .distributions import Distributions
from .errors import ErrorFunctions
from .fit_results import SingleFitResult

logger = logging.getLogger("los_estimator")

#: Relative step for the central-difference Hessian. ``eps**0.25`` is the
#: textbook optimum for a second derivative in double precision. Computed from
#: `sys.float_info` (not `np.finfo`) so this stays a plain float at import
#: time even when numpy is mocked out (e.g. Sphinx's `autodoc_mock_imports`).
_FD_REL_STEP = sys.float_info.epsilon**0.25

#: Below this the Hessian is treated as numerically zero (``sentinel``, whose
#: only parameter does not influence the kernel at all, hits this exactly).
_HESSIAN_ZERO_TOL = 1e-12


def combine_past_kernel(past_kernels: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Stack previously fitted kernels with the new one (row-wise)."""
    return np.vstack([*past_kernels, kernel])


def _fd_steps(x, bounds):
    """Central-difference step sizes, clipped to stay inside `bounds`.
    Returns None if a parameter sits on (or too near) a bound."""
    x = np.asarray(x, dtype=float)
    steps = _FD_REL_STEP * np.maximum(np.abs(x), 1.0)
    if bounds is not None:
        for i, bound in enumerate(bounds):
            low, high = bound
            if low is not None and np.isfinite(low):
                steps[i] = min(steps[i], (x[i] - low) / 2.0)
            if high is not None and np.isfinite(high):
                steps[i] = min(steps[i], (high - x[i]) / 2.0)
    if not np.all(np.isfinite(steps)) or np.any(steps <= 1e-12 * np.maximum(np.abs(x), 1.0)):
        return None
    return steps


def finite_difference_hessian(fun, x, bounds=None):
    """Symmetric Hessian of `fun` at `x` via central finite differences
    (~2*ndim**2 evaluations, bounds respected). None if it can't be computed."""
    x = np.asarray(x, dtype=float)
    ndim = x.size
    if ndim == 0:
        return None
    steps = _fd_steps(x, bounds)
    if steps is None:
        return None

    def f(point):
        return float(fun(point))

    f0 = f(x)
    if not np.isfinite(f0):
        return None

    hessian = np.zeros((ndim, ndim))
    for i in range(ndim):
        e_i = np.zeros(ndim)
        e_i[i] = steps[i]
        f_plus = f(x + e_i)
        f_minus = f(x - e_i)
        hessian[i, i] = (f_plus - 2.0 * f0 + f_minus) / steps[i] ** 2
        for j in range(i + 1, ndim):
            e_j = np.zeros(ndim)
            e_j[j] = steps[j]
            f_pp = f(x + e_i + e_j)
            f_pm = f(x + e_i - e_j)
            f_mp = f(x - e_i + e_j)
            f_mm = f(x - e_i - e_j)
            hessian[i, j] = hessian[j, i] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * steps[i] * steps[j])
    if not np.all(np.isfinite(hessian)):
        return None
    return 0.5 * (hessian + hessian.T)


def invert_loss_hessian(hessian):
    """Invert a loss Hessian, returning (inverse_or_None, reason). None when the
    Hessian is numerically zero, indefinite, or too ill-conditioned to invert --
    checked here since `np.random.multivariate_normal` silently accepts a
    non-PSD covariance."""
    if hessian is None:
        return None, "no hessian"
    hessian = np.atleast_2d(np.asarray(hessian, dtype=float))
    if not np.all(np.isfinite(hessian)):
        return None, "non-finite hessian"
    eigenvalues = np.linalg.eigvalsh(hessian)
    scale = float(np.max(np.abs(eigenvalues)))
    if scale <= _HESSIAN_ZERO_TOL:
        return None, "numerically zero hessian"
    if float(np.min(eigenvalues)) <= scale * 1e-10:
        return None, "hessian not positive definite"
    try:
        inverse = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        return None, "singular hessian"
    if not np.all(np.isfinite(inverse)):
        return None, "non-finite hessian inverse"
    return inverse, "ok"


def laplace_covariance(hessian_inv, mse, n_residuals):
    """Rescale an inverse loss Hessian into a Laplace posterior covariance:
    `minimize` optimizes MSE, not a negative log-likelihood, so the inverse
    Hessian must be scaled by 2*sigma_hat**2/n. None if inputs are degenerate."""
    if hessian_inv is None or n_residuals <= 0 or not np.isfinite(mse) or mse <= 0:
        return None
    covariance = (2.0 * float(mse) / float(n_residuals)) * np.asarray(hessian_inv, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    if not np.all(np.isfinite(covariance)):
        return None
    return covariance


#: Reasons for which falling back to the optimizer's own ``hess_inv`` is allowed.
_FD_UNAVAILABLE = frozenset({"no hessian", "non-finite hessian", "non-finite hessian inverse"})


def _dense_hess_inv(result):
    """Optimizer's inverse Hessian as a dense matrix (only L-BFGS-B provides one).
    Rejected if not positive definite, or a multiple of the identity -- the
    latter is what L-BFGS-B returns when it never took a successful step, which
    carries no curvature information and would produce a fabricated band."""
    raw = getattr(result, "hess_inv", None)
    if raw is None:
        return None
    try:
        dense = np.atleast_2d(np.asarray(raw.todense() if hasattr(raw, "todense") else raw, dtype=float))
    except Exception:  # pragma: no cover - optimizer specific
        return None
    if dense.size == 0 or not np.all(np.isfinite(dense)):
        return None
    dense = 0.5 * (dense + dense.T)
    if float(np.min(np.linalg.eigvalsh(dense))) <= 0:
        return None
    if np.allclose(dense, dense[0, 0] * np.eye(dense.shape[0])):
        return None
    return dense


def compute_covariance(distro, obj_fun, result, args, distro_boundaries, train_mse, n_residuals):
    """Laplace posterior covariance for a finished fit: central-difference Hessian
    first, falling back to the optimizer's own hess_inv (L-BFGS-B only), else None.
    Returns (covariance, source_description)."""
    hessian = finite_difference_hessian(lambda p: obj_fun(p, *args), result.x, distro_boundaries)
    hessian_inv, reason = invert_loss_hessian(hessian)
    source = "finite-difference"
    if hessian_inv is None and reason in _FD_UNAVAILABLE:
        # Only when the finite differences could not be *evaluated*. A Hessian
        # that was computed and found flat or indefinite is a statement about the
        # loss surface, and the limited-memory ``hess_inv`` does not get to
        # overrule it: it would silently manufacture a band for an
        # unidentifiable parameter (``sentinel``'s stretch, for example).
        dense = _dense_hess_inv(result)
        if dense is not None:
            hessian_inv = dense
            source = f"hess_inv fallback ({reason})"
    if hessian_inv is None:
        logger.debug(f"{distro}: no usable covariance ({reason})")
        return None, f"none ({reason})"

    covariance = laplace_covariance(hessian_inv, train_mse, n_residuals)
    if covariance is None:
        logger.debug(f"{distro}: covariance scaling failed (mse={train_mse}, n={n_residuals})")
        return None, "none (scaling failed)"
    logger.debug(f"{distro}: covariance from {source}")
    return covariance, source


def get_objective_convolution(
    distro: str, kernel_width: int, error_fun: Callable[[np.ndarray, np.ndarray], float]
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]], float]:
    """Build an objective closure: generate a kernel for `distro`, convolve, score with `error_fun`."""

    def objective_function(
        model_config: np.ndarray,
        inc: np.ndarray,
        icu: np.ndarray,
        past_kernels: Optional[np.ndarray] = None,
        return_prediction=False,
    ) -> float:
        kernel = Distributions.generate_kernel(distro, model_config, kernel_width)
        if past_kernels is not None:
            kernel = combine_past_kernel(past_kernels, kernel)

        observed = calc_its_convolution(inc, kernel)
        res = error_fun(icu[kernel_width:], observed[kernel_width:])
        if return_prediction:
            return res, observed
        return res

    return objective_function


def initialize_distro_parameters(
    distro: str,
    distro_boundaries: Optional[List[Tuple[Optional[float], Optional[float]]]],
    distro_init_params: Optional[List[float]],
) -> Tuple[List[Tuple[Optional[float], Optional[float]]], List[float]]:
    """Fill in default boundaries/init params from the distribution table if not
    given. The trailing scaling_fac is appended only when `uses_scaling` is set."""

    distribution = Distributions[distro]

    if distro_boundaries is None:
        distro_boundaries = list(distribution.boundaries)
        if distribution.uses_scaling:
            distro_boundaries = distro_boundaries + [(None, None)]

    if distro_init_params is None or len(distro_init_params) == 0:
        distro_init_params = list(distribution.init_values)
        if distribution.uses_scaling:
            distro_init_params = distro_init_params + [distribution.scaling_init]

    return distro_boundaries, distro_init_params


def fit_convolution(
    distro: str,
    train_data: Tuple[np.ndarray, np.ndarray],
    test_data: Tuple[np.ndarray, np.ndarray],
    kernel_width: int,
    distro_boundaries: Optional[List[Tuple[Optional[float], Optional[float]]]] = None,
    distro_init_params: Optional[List[float]] = None,
    past_kernels: Optional[np.ndarray] = None,
    method: str = "L-BFGS-B",
    error_fun: str = "mse",
) -> SingleFitResult:
    """Fit `distro`'s kernel to (train_data) via `scipy.optimize.minimize`,
    then evaluate on train/test and estimate the Laplace covariance."""

    error_fun = ErrorFunctions[error_fun]

    distro_boundaries, distro_init_params = initialize_distro_parameters(distro, distro_boundaries, distro_init_params)

    obj_fun = get_objective_convolution(distro, kernel_width, error_fun)

    args = (
        *train_data,
        past_kernels,
    )

    result = minimize(
        obj_fun,
        x0=distro_init_params,
        args=args,
        bounds=distro_boundaries,
        method=method,
    )

    distro_params = result.x

    fitted_kernel = Distributions.generate_kernel(distro, distro_params, kernel_width)

    train_err, train_prediction = obj_fun(distro_params, *train_data, past_kernels, return_prediction=True)
    test_err, test_prediction = obj_fun(distro_params, *test_data, past_kernels, return_prediction=True)

    rel_train_error = train_prediction[kernel_width:] / train_data[1][kernel_width:]
    rel_test_error = test_prediction[kernel_width:] / test_data[1][kernel_width:]

    # Laplace covariance at the optimum. The MSE is recomputed explicitly here so
    # that the noise scale stays correct even when ``error_fun`` is not ``mse``.
    residuals = np.asarray(train_data[1][kernel_width:]) - np.asarray(train_prediction[kernel_width:])
    train_mse = float(np.mean(residuals**2)) if residuals.size else float("nan")
    n_residuals = int(len(train_data[0]) - kernel_width)
    covariance, covariance_source = compute_covariance(
        distro=distro,
        obj_fun=obj_fun,
        result=result,
        args=args,
        distro_boundaries=distro_boundaries,
        train_mse=train_mse,
        n_residuals=n_residuals,
    )

    fit_results = SingleFitResult(
        distro=distro,
        train_data=train_data,
        test_data=test_data,
        success=result.success,
        minimization_result=result,
        train_error=train_err,
        test_error=test_err,
        kernel=fitted_kernel,
        train_prediction=train_prediction,
        test_prediction=test_prediction,
        distro_params=distro_params,
        rel_train_error=rel_train_error,
        rel_test_error=rel_test_error,
        covariance=covariance,
        covariance_source=covariance_source,
        train_mse=train_mse,
        n_residuals=n_residuals,
    )

    return fit_results


def objective_compartemental(
    error_fun: Callable[[np.ndarray, np.ndarray], float],
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, int], float]:
    """Build an objective closure scoring the compartmental prediction with `error_fun`."""

    def objective_function(model_config: np.ndarray, inc: np.ndarray, icu: np.ndarray, kernel_width: int) -> float:
        discharge_rate, transition_rate, delay = model_config
        pred = calc_its_comp(inc, discharge_rate, transition_rate, delay, init=icu[0])
        return error_fun(pred[kernel_width : len(icu)], icu[kernel_width:])

    return objective_function


def fit_compartmental(
    train_data: Tuple[np.ndarray, np.ndarray],
    test_data: Tuple[np.ndarray, np.ndarray],
    initial_guess_comp: List[float],
    kernel_width: int,
    method: str = "TNC",
    error_fun: str = "mse",
) -> SingleFitResult:
    """Fit the compartmental model to a train/test split via bounded optimization."""
    x_train, y_train = train_data
    x_test, y_test = test_data

    error_fun = ErrorFunctions[error_fun]
    obj_fun = objective_compartemental(error_fun)

    result = minimize(
        obj_fun,
        initial_guess_comp,
        args=(x_train, y_train, kernel_width),
        method=method,
        bounds=[(0, 1), (1, 1), (0, 0)],
    )
    train_prediction = calc_its_comp(x_train, *result.x, y_train[0])
    test_prediction = calc_its_comp(x_test, *result.x, y_test[0])

    train_err = obj_fun(result.x, x_train, y_train, kernel_width)
    test_err = obj_fun(result.x, x_test, y_test, kernel_width)

    rel_train_error = train_prediction / train_data[1]
    rel_test_error = test_prediction / test_data[1]

    result_obj = SingleFitResult(
        distro="compartmental",
        train_data=x_train,
        test_data=x_test,
        success=result.success,
        minimization_result=result,
        train_error=train_err,
        test_error=test_err,
        kernel=np.zeros(1),
        train_prediction=train_prediction,
        test_prediction=test_prediction,
        distro_params=result.x,
        rel_train_error=rel_train_error,
        rel_test_error=rel_test_error,
    )

    return result_obj
