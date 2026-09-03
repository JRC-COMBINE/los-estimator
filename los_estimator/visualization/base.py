"""Base visualizer class with common functionality."""

import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt

from ..config import OutputFolderConfig, VisualizationConfig


def get_color_palette() -> List[str]:
    """Matplotlib's default color cycle extended with extra hex colors, for
    plots needing more distinct series than the default cycle provides."""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors += [
        "#FFA07A",
        "#20B2AA",
        "#FF6347",
        "#808000",
        "#FF00FF",
        "#FFD700",
        "#00FF00",
        "#00FFFF",
        "#0000FF",
        "#8A2BE2",
    ]
    return colors


class VisualizerBase:
    """Common plot styling, color management, and file saving for visualizers."""

    def __init__(
        self,
        visualization_config: VisualizationConfig,
        output_config: Optional[OutputFolderConfig] = None,
    ):
        self.visualization_config: VisualizationConfig = visualization_config
        self.output_config: Optional[OutputFolderConfig] = output_config
        if output_config is not None:
            self.output_path = output_config.figures

        try:
            plt.style.use(visualization_config.style)
        except OSError:
            plt.style.use("default")

        self.figsize: Tuple[float, float] = visualization_config.figsize
        self.colors: List[str] = visualization_config.colors

        # Set high-quality defaults
        plt.rcParams["savefig.facecolor"] = visualization_config.savefig_facecolor
        plt.rcParams["savefig.dpi"] = visualization_config.savefig_dpi
        plt.rcParams["figure.dpi"] = 100

    def _figure(self, *args, **kwargs) -> plt.Figure:
        """Create a new figure with specified size and DPI."""
        figsize = kwargs.pop("figsize", self.figsize)
        plt.ioff()
        return plt.figure(*args, figsize=figsize, **kwargs)

    def _get_subplots(self, *args, **kwargs) -> Tuple[plt.Figure, List[plt.Axes]]:
        """Create subplots with specified number of rows and columns."""
        figsize = kwargs.pop("figsize", self.figsize)
        return plt.subplots(*args, figsize=figsize, **kwargs)

    def _show(self, filename: Optional[str] = None, fig: Optional[plt.Figure] = None):
        """Save (using `visualization_config.file_format`, overriding any
        extension already in `filename`) and/or show the figure."""
        if fig is None:
            fig = plt.gcf()

        if self.visualization_config.save_figures:
            if filename and self.output_config:
                ext = "." + self.visualization_config.file_format.lstrip(".")
                base, _ = os.path.splitext(filename)
                filename = base + ext
                full_path = os.path.join(self.output_path, filename)
                fig.savefig(full_path, bbox_inches="tight")

        if self.visualization_config.show_figures:
            plt.show(block=False)
            plt.pause(0.001)
            plt.show()
        else:
            plt.close(fig)
