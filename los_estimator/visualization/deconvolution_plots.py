# %%
"""Deconvolution plotting functionality."""

import logging
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
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
    """Plotting functionality for deconvolution analysis.

    Provides comprehensive visualization capabilities for analyzing and
    presenting the results of length of stay deconvolution models, including
    fit comparisons, kernel visualizations, and error analysis plots.

    Attributes:
        all_fit_results (MultiSeriesFitResults): Container with all fitting results.
        series_data (SeriesData): Time series data used for fitting.
        model_config (ModelConfig): Configuration for model parameters.
        visualization_context (VisualizationContext): Shared visualization context.
        output_config (OutputFolderConfig): Output directory configuration.
    """

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

    def _ax_plot_error_error_points(self, ax2, fr_series, distro):
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
        ax2.set_xlabel("Days")

    def show_error_windows(self, distro: Optional[Union[str, List[str]]] = None):
        """Show error windows for specified distributions."""
        distros = self._get_distro_as_array(distro)

        for distro in distros:
            fr_series = self.all_fit_results[distro]
            _, (ax, ax2) = self._get_subplots(2, 1, sharex=True, figsize=(12, 6))
            self._ax_plot_prediction_error_window(ax, fr_series, distro)
            self._ax_plot_error_error_points(ax2, fr_series, distro)

            plt.suptitle(f"{distro.capitalize()} Distribution\n{self.model_config.run_name}")
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

    def superimpose_kernels(self, distro: Optional[Union[str, List[str]]] = None):
        """Show superimposed kernels for distributions."""
        distros = self._get_distro_as_array(distro)

        for distro in distros:
            self._figure(figsize=(10, 5))

            # plot real kernel
            l, r = None, None
            if self.vc.real_los is not None:
                (r,) = plt.plot(self.vc.real_los, color="black", label="Sample Kernel")

            fit_results = self.all_fit_results[distro]
            for fit_result in fit_results.fit_results:
                (l,) = plt.plot(
                    fit_result.kernel,
                    alpha=0.3,
                    color=self.colors[0],
                    label=f"Rolling {distro.capitalize()} Kernels",
                )
            handles = []
            if r is not None:
                handles.append(r)
            if l is not None:
                handles.append(l)
            plt.legend(handles=handles)
            plt.ylim(-0.005, 0.3)
            plt.xlim(-1, self.model_config.kernel_width + 1)
            plt.xlabel("Days after admission")
            plt.ylabel("Discharge Probability")

            self._set_title(f"All Rolling {distro.capitalize()} Kernels")
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
        _, axs = self._get_subplots(max(1, n_distros // 3), 3, sharex=True, sharey=True, figsize=(12, 6))
        axs = axs.flatten()
        for distro, ax in zip(self.all_fit_results.distros, axs):
            fr = self.all_fit_results[distro]
            x = fr.train_errors
            y = fr.test_errors
            ax.scatter(x, y, s=10)
            ax.set_xlabel(f"Train {self.error_fun.capitalize()}")
            ax.set_ylabel(f"Test {self.error_fun.capitalize()}")
            ax.set_title(f"{distro.capitalize()} Distribution")
        plt.suptitle(self._get_full_title("Train vs Test Error"))
        plt.tight_layout()
        self._show(f"train_vs_test_error.png")

    def _get_full_title(self, title: str) -> str:
        """Get full title with run name."""
        run_name = self.model_config.run_name
        return title + "\n" + run_name

    def _set_title(self, title: str, *args, **kwargs):
        """Set the title of the current figure."""
        plt.title(self._get_full_title(title), *args, **kwargs)
