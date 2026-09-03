# %%
"""Deconvolution plotting functionality."""

import logging
import math
import os
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from ..config import (
    ModelConfig,
    OutputFolderConfig,
    VisualizationConfig,
    VisualizationContext,
)
from ..core import SeriesData
from ..fitting import MultiSeriesFitResults
from .base import VisualizerBase

logger = logging.getLogger("los_estimator")


# %%


class DeconvolutionPlots(VisualizerBase):
    """Fit comparisons, kernel visualizations, and error-analysis plots for a run."""

    def __init__(
        self,
        all_fit_results: MultiSeriesFitResults,
        series_data: SeriesData,
        model_config: ModelConfig,
        visualization_config: VisualizationConfig,
        visualization_context: VisualizationContext,
        output_config: OutputFolderConfig,
    ):
        super().__init__(visualization_config, output_config)

        self.vc: VisualizationContext = visualization_context
        self.all_fit_results: MultiSeriesFitResults = all_fit_results
        self.series_data: SeriesData = series_data
        self.model_config: ModelConfig = model_config

        self.error_fun = model_config.error_fun

    def _pairplot(self, col2, col1):
        """Create scatter plot comparing two metrics."""
        name = f"{col2}_vs_{col1}"

        fig = self._figure()

        for i, distro in enumerate(self.all_fit_results.distros):
            if distro in ["sentinel"]:
                continue
            val1 = self.all_fit_results.summary[col1][distro]
            val2 = self.all_fit_results.summary[col2][distro]
            plt.scatter(val1, val2, s=100, label=distro, color=self.colors[i])
            plt.annotate(
                distro,
                (val1, val2),
                fontsize=9,
                xytext=(5, 5),
                textcoords="offset points",
            )

        # Labels and formatting
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.grid(True)
        self._set_title(f"Model Performance: {name}")
        self._show(name or f"{col2}_vs_{col1}.png", fig)

    def plot_error_comparison(self):
        """Plot error comparison across models."""
        sorted_summary = self.all_fit_results.summary.sort_values("Median Loss Train")
        sorted_summary = sorted_summary[
            [
                # "Mean Loss Train",
                "Median Loss Train",
                # "Upper Quartile Train",
                # "Lower Quartile Train",
                # "Mean Loss Test",
                "Median Loss Test",
                "Mean Loss Test (no outliers)",
                "Mean Loss Train (no outliers)",
            ]
        ]

        sorted_summary.plot.bar(subplots=False, figsize=(10, 6))

        plt.legend()
        xticks = list(sorted_summary.index)
        plt.xticks(np.arange(len(xticks)), xticks, rotation=45)
        plt.xlabel("Distribution Functions")
        plt.ylabel(self.error_fun.capitalize())
        plt.suptitle(self._get_full_title("Error Comparison of Models"))
        plt.tight_layout()
        self._show("error_comparison.png")

    def boxplot_errors(self, errors, title, ylabel, file, show_outliers):
        """Create boxplot of errors."""
        self._figure()
        plt.boxplot(errors, showfliers=show_outliers)
        distro_and_n = [f"{distro.capitalize()}" for distro, fr in self.all_fit_results.items()]
        plt.xticks(np.arange(len(distro_and_n)) + 1, distro_and_n, rotation=45)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.tight_layout()
        self._show(file)

    def _ax_plot_prediction_error_window(self, ax, fr_series, distro):
        """Plot prediction error window on given axis."""
        (l_real,) = ax.plot(
            self.series_data.y_full,
            color="black",
            alpha=0.8,
            linestyle="--",
            label="Real Occupancy",
        )
        for w, fit_result in zip(fr_series.window_infos, fr_series.fit_results):

            x = np.arange(w.training_prediction_start, w.train_end)
            y = fit_result.train_prediction[self.model_config.kernel_width : self.model_config.train_width]
            (l_train,) = ax.plot(
                x,
                y,
                color=self.colors[0],
                label=f"{distro.capitalize()} Train",
                linestyle="-",
            )
            if fit_result.train_lower is not None and fit_result.train_upper is not None:
                lo = fit_result.train_lower[self.model_config.kernel_width : self.model_config.train_width]
                hi = fit_result.train_upper[self.model_config.kernel_width : self.model_config.train_width]
                ax.fill_between(x, lo, hi, color=self.colors[0], alpha=0.2, linewidth=0, zorder=1)

            x = np.arange(w.train_end, w.test_end)
            y = fit_result.test_prediction[w.kernel_width : w.kernel_width + self.model_config.test_width]
            (l_test,) = ax.plot(
                x,
                y,
                color=self.colors[1],
                label=f"{distro.capitalize()} Prediction",
                linestyle="--",
                alpha=0.5,
            )
            if fit_result.test_lower is not None and fit_result.test_upper is not None:
                lo = fit_result.test_lower[w.kernel_width : w.kernel_width + self.model_config.test_width]
                hi = fit_result.test_upper[w.kernel_width : w.kernel_width + self.model_config.test_width]
                ax.fill_between(x, lo, hi, color=self.colors[1], alpha=0.2, linewidth=0, zorder=1)
        legend_handles = [l_real, l_train, l_test]
        [
            plt.Line2D([0], [0], color="black", linestyle="--", label="Real"),
            plt.Line2D([0], [0], color=self.colors[0], label=f"{distro.capitalize()} Train"),
            plt.Line2D(
                [0],
                [0],
                color=self.colors[1],
                label=f"{distro.capitalize()} Prediction",
            ),
        ]
        ax.legend(handles=legend_handles, loc="upper right")

        ax.set_ylim(-100, 6000)
        ax.set_xticks(self.vc.xtick_pos[1::2])
        ax.set_xticklabels(self.vc.xtick_label[1::2])
        ax.set_xlim(*self.vc.xlims)
        ax.set_ylabel("Ouccupied Beds")
        ax.set_title("Predictions vs Real Occupancy")
        ax.grid(zorder=0)

    def _ax_plot_error_error_points(self, ax2, fr_series, distro, include_xlabel: bool = True):
        """Plot error points on given axis."""
        x = self.series_data.windows
        (l1,) = ax2.plot(
            x,
            fr_series.train_errors,
            label="Train Error",
            color=self.colors[0],
            linestyle="-",
            alpha=0.7,
        )
        (l2,) = ax2.plot(
            x,
            fr_series.test_errors,
            label="Test Error",
            color=self.colors[1],
            linestyle="--",
            alpha=0.7,
        )

        ax2.legend(handles=[l1, l2], loc="upper right")

        ax2.set_title("Rolling Fit Errors")
        ax2.grid(zorder=0)
        ax2.set_ylabel(self.error_fun.capitalize())
        if include_xlabel:
            ax2.set_xlabel("Days")
        # Error magnitudes (e.g. MSE) are commonly large; a fixed power-of-ten
        # scaling factor (shown as the axis offset text) keeps tick labels
        # short instead of printing every digit.
        ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    def show_error_windows(
        self,
        distro: Optional[Union[str, List[str]]] = None,
        include_run_name: bool = True,
        include_title: bool = True,
        include_xlabel: bool = True,
    ):
        """Show error windows for specified distributions (defaults to all)."""
        distros = self._get_distro_as_array(distro)

        for distro in distros:
            fr_series = self.all_fit_results[distro]
            _, (ax, ax2) = self._get_subplots(2, 1, sharex=True, figsize=(12, 6))
            self._ax_plot_prediction_error_window(ax, fr_series, distro)
            self._ax_plot_error_error_points(ax2, fr_series, distro, include_xlabel=include_xlabel)

            if include_title:
                title = f"{distro.capitalize()} Distribution"
                if include_run_name:
                    title += f"\n{self.model_config.run_name}"
                plt.suptitle(title)
            plt.tight_layout()
            self._show(f"prediction_error_{distro}_fit.png")

    def show_all_error_windows_superimposed(self):
        """Show all error windows superimposed."""
        _, (ax, ax2) = self._get_subplots(2, 1, sharex=True, figsize=(12, 6))
        for distro in self.all_fit_results.distros:
            fr_series = self.all_fit_results[distro]

            self._ax_plot_prediction_error_window(ax, fr_series, distro)
            self._ax_plot_error_error_points(ax2, fr_series, distro)
        plt.suptitle(self._get_full_title("All Predictions and Error"))
        plt.tight_layout()
        self._show("prediction_error_all_distros.png")

    def _get_distro_as_array(self, distro: Optional[Union[str, List[str]]] = None) -> List[str]:
        """Convert distro parameter to array format."""
        if distro is None:
            distros = self.all_fit_results.distros
        elif isinstance(distro, str):
            distros = [distro]
        else:
            distros = distro
        return distros

    def show_all_predictions(self):
        """Show all predictions together."""
        _, ax = self._get_subplots(1, 1, sharex=True, figsize=(15, 7.5))
        for distro in self.all_fit_results.distros:
            fr_series = self.all_fit_results[distro]

            self._ax_plot_prediction_error_window(ax, fr_series, distro)

        ax.legend(labels=["Real Occupancy", "Train", "Test"])
        self._set_title("All Predictions")
        plt.tight_layout()
        self._show("prediction_all_distros.png")

    def superimpose_kernels(
        self,
        distro: Optional[Union[str, List[str]]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        include_run_name: bool = True,
        figsize: Tuple[float, float] = (10, 5),
        ylabel_fontsize: Optional[float] = None,
    ):
        """Show superimposed kernels for distributions (defaults to all).

        `ylim=None` auto-scales from the max plotted value (kernel, its band,
        and the reference kernel) instead of a fixed ceiling that leaves small
        kernels' plots mostly empty. `ylabel_fontsize` overrides just the
        "Discharge Probability" label, useful when a caller-drawn overlay
        crowds the default-size label on a short, wide figure.
        """
        distros = self._get_distro_as_array(distro)

        for distro in distros:
            self._figure(figsize=figsize)

            # plot real kernel
            l, r = None, None
            if self.vc.real_los is not None:
                (r,) = plt.plot(self.vc.real_los, color="black", label="Sample Kernel")

            fit_results = self.all_fit_results[distro]
            data_max = 0.0
            if self.vc.real_los is not None:
                data_max = max(data_max, float(np.max(self.vc.real_los)))
            for fit_result in fit_results.fit_results:
                if fit_result.kernel_lower is not None and fit_result.kernel_upper is not None:
                    plt.fill_between(
                        np.arange(len(fit_result.kernel)),
                        fit_result.kernel_lower,
                        fit_result.kernel_upper,
                        color=self.colors[0],
                        alpha=0.1,
                        linewidth=0,
                        zorder=1,
                    )
                    data_max = max(data_max, float(np.max(fit_result.kernel_upper)))
                (l,) = plt.plot(
                    fit_result.kernel,
                    alpha=0.3,
                    color=self.colors[0],
                    label=f"Rolling {distro.capitalize()} Kernels",
                    zorder=2,
                )
                data_max = max(data_max, float(np.max(fit_result.kernel)))
            handles = []
            if r is not None:
                handles.append(r)
            if l is not None:
                handles.append(l)
            plt.legend(handles=handles)
            if ylim is not None:
                plt.ylim(*ylim)
            else:
                data_max = data_max * 1.1 if data_max > 0 else 0.3
                plt.ylim(-0.02 * data_max, data_max)
            plt.xlim(-1, self.model_config.kernel_width + 1)
            plt.xlabel("Days after admission")
            plt.ylabel("Discharge Probability", fontsize=ylabel_fontsize)

            title = f"All Rolling {distro.capitalize()} Kernels"
            if include_run_name:
                self._set_title(title)
            else:
                plt.title(title)
            plt.tight_layout()
            plt.grid()
            self._show(f"rolling_kernels_{distro}.png")

    def generate_plots_for_run(self):
        """Generate all plots for a run."""

        self.plot_error_comparison()
        self.boxplot_errors(
            self.all_fit_results.train_errors_by_distro,
            "Train Error",
            self.error_fun.capitalize(),
            "train_error_boxplot.png",
            show_outliers=True,
        )
        self.boxplot_errors(
            self.all_fit_results.train_errors_by_distro,
            "Train Error",
            self.error_fun.capitalize(),
            "train_error_boxplot_no_outliers.png",
            show_outliers=False,
        )
        self.boxplot_errors(
            self.all_fit_results.test_errors_by_distro,
            "Test Error",
            self.error_fun.capitalize(),
            "test_error_boxplot.png",
            show_outliers=True,
        )
        self.boxplot_errors(
            self.all_fit_results.test_errors_by_distro,
            "Test Error",
            self.error_fun.capitalize(),
            "test_error_boxplot_no_outliers.png",
            show_outliers=False,
        )

        self.show_error_windows()
        self.show_all_error_windows_superimposed()
        self.show_all_predictions()
        self.superimpose_kernels()

        self.plot_train_vs_test_error()

    def plot_train_vs_test_error(self):
        n_distros = len(self.all_fit_results.distros)
        n_cols = 3
        n_rows = math.ceil(n_distros / n_cols)
        _, axs = self._get_subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=(12, 4 * n_rows))
        axs = axs.flatten()
        for distro, ax in zip(self.all_fit_results.distros, axs):
            fr = self.all_fit_results[distro]
            x = fr.train_errors
            y = fr.test_errors
            ax.scatter(x, y, s=10)
            ax.set_xlabel(f"Train {self.error_fun.capitalize()}")
            ax.set_ylabel(f"Test {self.error_fun.capitalize()}")
            ax.set_title(f"{distro.capitalize()} Distribution")
        for ax in axs[n_distros:]:
            ax.axis("off")
        plt.suptitle(self._get_full_title("Train vs Test Error"))
        plt.tight_layout()
        self._show(f"train_vs_test_error.png")

    def _ax_plot_coverage(self, ax, distro, coverage_detail: pd.DataFrame):
        """Plot empirical coverage vs. nominal coverage for one distribution."""
        sub = coverage_detail[coverage_detail["distribution"] == distro]
        sub = sub.dropna(subset=["coverage"]).sort_values("window")
        nominal = float(coverage_detail["nominal_coverage"].iloc[0])

        ax.plot(
            sub["window"],
            sub["coverage"],
            color=self.colors[2 % len(self.colors)],
            marker="o",
            markersize=3,
            label="Empirical coverage",
        )
        ax.axhline(
            nominal,
            color="black",
            linestyle="--",
            alpha=0.7,
            label=f"nominal ({nominal:.0%})",
        )
        ax.set_ylabel("Empirical coverage")
        ax.set_xlabel("Days")
        ax.set_title("Test-band coverage over time")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True)

    def show_coverage_dashboards(self, coverage_detail: pd.DataFrame):
        """Save one figure per distribution (rolling fit/band on top, coverage
        below) to `figures/coverage/<distro>_coverage_dashboard.png`. Skips
        distributions with no windowed UQ band."""
        valid = coverage_detail.dropna(subset=["coverage"])
        if valid.empty:
            logger.info("No windows carry a UQ band; skipping coverage dashboards")
            return

        coverage_dir = os.path.join(self.output_config.figures, "coverage")
        os.makedirs(coverage_dir, exist_ok=True)
        original_output_path = self.output_path
        self.output_path = coverage_dir
        try:
            for distro in sorted(valid["distribution"].unique()):
                fr_series = self.all_fit_results[distro]
                fig, (ax_fit, ax_cov) = self._get_subplots(2, 1, sharex=True, figsize=(12, 8))
                self._ax_plot_prediction_error_window(ax_fit, fr_series, distro)
                self._ax_plot_coverage(ax_cov, distro, coverage_detail)
                plt.suptitle(f"{distro.capitalize()} Distribution — Fit & Coverage\n" f"{self.model_config.run_name}")
                plt.ylim(-0.05, None)
                plt.tight_layout()
                self._show(f"{distro}_coverage_dashboard.png", fig)
        finally:
            self.output_path = original_output_path

    def _get_full_title(self, title: str) -> str:
        """Get full title with run name."""
        run_name = self.model_config.run_name
        return title + "\n" + run_name

    def _set_title(self, title: str, *args, **kwargs):
        """Set the title of the current figure."""
        plt.title(self._get_full_title(title), *args, **kwargs)
