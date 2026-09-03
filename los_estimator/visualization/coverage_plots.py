"""Width-and-coverage-over-time figure for the UQ percentile bands."""

import logging

import matplotlib.pyplot as plt
import pandas as pd

from ..config import OutputFolderConfig, VisualizationConfig
from .base import VisualizerBase

logger = logging.getLogger("los_estimator")


class CoveragePlots(VisualizerBase):
    """Plot empirical coverage and band width over time, per distribution.
    `detail` is the per-window table from `evaluation.coverage.compute_coverage_table`."""

    def __init__(
        self,
        detail: pd.DataFrame,
        visualization_config: VisualizationConfig,
        output_config: OutputFolderConfig,
    ):
        super().__init__(visualization_config, output_config)
        self.detail: pd.DataFrame = detail
        # save alongside the other coverage tables, not the figures/ folder
        self.output_path = output_config.metrics

    def plot_coverage_over_time(
        self,
        include_band_width: bool = True,
        xtick_pos=None,
        xtick_label=None,
    ):
        """Save `coverage_over_time.png`: coverage (and, by default, band width) vs. window.

        `xtick_pos`/`xtick_label` (both or neither) let this share the same
        month/year ticks as the prediction plots (e.g. `VisualizationContext.xtick_pos`)
        instead of raw window numbers.
        """
        valid = self.detail.dropna(subset=["coverage"])
        if valid.empty:
            logger.info("No windows carry a UQ band; skipping coverage_over_time.png")
            return

        distros = sorted(valid["distribution"].unique())
        nominal = float(self.detail["nominal_coverage"].iloc[0])

        if include_band_width:
            fig, (ax_cov, ax_width) = self._get_subplots(2, 1, sharex=True, figsize=(12, 7))
        else:
            fig, ax_cov = self._get_subplots(1, 1, figsize=(12, 4.5))

        for i, distro in enumerate(distros):
            sub = valid[valid["distribution"] == distro].sort_values("window")
            color = self.colors[i % len(self.colors)]
            ax_cov.plot(
                sub["window"],
                sub["coverage"],
                label=distro,
                color=color,
                marker="o",
                markersize=3,
            )
            if include_band_width:
                ax_width.plot(
                    sub["window"],
                    sub["band_width"],
                    label=distro,
                    color=color,
                    marker="o",
                    markersize=3,
                )

        ax_cov.axhline(
            nominal,
            color="black",
            linestyle="--",
            alpha=0.7,
            label=f"nominal ({nominal:.0%})",
        )
        ax_cov.set_ylabel("Empirical coverage")
        ax_cov.set_title("Test-band coverage over time")
        ax_cov.legend(loc="lower left")
        ax_cov.grid(True)

        use_date_ticks = xtick_pos is not None and xtick_label is not None
        if use_date_ticks:
            ax_cov.set_xticks(xtick_pos)
            ax_cov.set_xticklabels(xtick_label)

        if include_band_width:
            ax_width.set_ylabel("Mean band width")
            ax_width.set_title("Test-band width over time")
            ax_width.grid(True)
            if use_date_ticks:
                ax_width.set_xticks(xtick_pos)
                ax_width.set_xticklabels(xtick_label)
            else:
                ax_width.set_xlabel("Window")
        elif not use_date_ticks:
            ax_cov.set_xlabel("Window")

        plt.tight_layout()
        self._show("coverage_over_time.png", fig)
