"""Empirical coverage validation for the UQ percentile bands: for every window
with a test band, checks `mean((y_test[kw:] >= lo) & (y_test[kw:] <= hi))`
against the nominal coverage from `confidence_interval` (e.g. 5/95 -> 0.90).

Windows without a band (failed fit, singular Hessian, ...) stay as NaN rows in
the detail table. `summarize_coverage` reports two numbers from them:
`mean_coverage` averages only over windows that got a band (blind to how often
a band was produced at all); `mean_coverage_effective` counts band-less windows
as a miss (coverage 0), so a distribution that only bands its easy windows
can't look well-calibrated on `mean_coverage` alone -- `band_rate` and
`mean_coverage_effective` surface that instead.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd

from los_estimator.fitting.fit_results import MultiSeriesFitResults

__all__ = [
    "compute_window_coverage",
    "compute_coverage_table",
    "summarize_coverage",
    "save_coverage_tables",
]


def compute_window_coverage(fit_result, kernel_width: int) -> Tuple[float, float]:
    """(coverage, mean_width) for one window's test band, or (None, None) if no band."""
    if (
        fit_result is None
        or fit_result.test_lower is None
        or fit_result.test_upper is None
    ):
        return None, None
    _, y_test = fit_result.test_data
    y = np.asarray(y_test, dtype=float)[kernel_width:]
    lo = np.asarray(fit_result.test_lower, dtype=float)[kernel_width:]
    hi = np.asarray(fit_result.test_upper, dtype=float)[kernel_width:]
    n = min(y.size, lo.size, hi.size)
    if n == 0:
        return None, None
    y, lo, hi = y[:n], lo[:n], hi[:n]
    coverage = float(np.mean((y >= lo) & (y <= hi)))
    width = float(np.mean(hi - lo))
    return coverage, width


def compute_coverage_table(
    all_fit_results: MultiSeriesFitResults,
    kernel_width: int,
    confidence_interval: Tuple[float, float],
) -> pd.DataFrame:
    """One row per (distribution, window): coverage, band_width (NaN if no band),
    and nominal_coverage."""
    nominal = (float(confidence_interval[1]) - float(confidence_interval[0])) / 100.0
    rows = []
    for distro, series in all_fit_results.items():
        for w, fr in zip(series.window_infos, series.fit_results):
            coverage, width = compute_window_coverage(fr, kernel_width)
            rows.append(
                {
                    "distribution": distro,
                    "window": w.window,
                    "coverage": np.nan if coverage is None else coverage,
                    "band_width": np.nan if width is None else width,
                    "nominal_coverage": nominal,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "distribution",
            "window",
            "coverage",
            "band_width",
            "nominal_coverage",
        ],
    )


def summarize_coverage(detail: pd.DataFrame) -> pd.DataFrame:
    """Per-distribution mean_coverage / mean_coverage_effective / band_rate etc.
    (see module docstring). A distribution with no band still appears, with
    mean_coverage NaN and mean_coverage_effective 0.0."""
    if detail.empty:
        return pd.DataFrame(
            columns=[
                "distribution",
                "mean_coverage",
                "mean_coverage_effective",
                "mean_band_width",
                "n_windows_with_band",
                "n_windows_total",
                "band_rate",
                "nominal_coverage",
            ]
        )

    nominal = float(detail["nominal_coverage"].iloc[0])
    valid = detail.dropna(subset=["coverage"])
    counts = valid.groupby("distribution")["coverage"].count()
    means = valid.groupby("distribution").agg(
        mean_coverage=("coverage", "mean"), mean_band_width=("band_width", "mean")
    )
    totals = detail.groupby("distribution")["window"].count()
    # band-less windows count as a miss (coverage 0) toward the effective mean
    effective = (
        detail.assign(coverage_or_miss=detail["coverage"].fillna(0.0))
        .groupby("distribution")["coverage_or_miss"]
        .mean()
    )

    summary = pd.DataFrame(index=sorted(detail["distribution"].unique()))
    summary["mean_coverage"] = means["mean_coverage"]
    summary["mean_coverage_effective"] = effective
    summary["mean_band_width"] = means["mean_band_width"]
    summary["n_windows_with_band"] = counts
    summary["n_windows_with_band"] = (
        summary["n_windows_with_band"].fillna(0).astype(int)
    )
    summary["n_windows_total"] = totals.astype(int)
    summary["band_rate"] = summary["n_windows_with_band"] / summary["n_windows_total"]
    summary["nominal_coverage"] = nominal
    summary.index.name = "distribution"
    return summary.reset_index()


def save_coverage_tables(
    detail: pd.DataFrame, summary: pd.DataFrame, path: str
) -> None:
    """Write summary/detail to `<path>/ci_coverage.csv` and `ci_coverage_detail.csv`."""
    os.makedirs(path, exist_ok=True)
    summary.to_csv(os.path.join(path, "ci_coverage.csv"), index=False)
    detail.to_csv(os.path.join(path, "ci_coverage_detail.csv"), index=False)
