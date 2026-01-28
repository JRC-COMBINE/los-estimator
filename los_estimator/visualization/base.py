"""Base visualizer class with common functionality."""

import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt

from ..config import OutputFolderConfig, VisualizationConfig


def get_color_palette() -> List[str]:
    """Get extended color palette for plotting.

    Returns a comprehensive color palette combining matplotlib's default
    color cycle with additional colors for extensive plotting needs.

    Returns:
        List[str]: List of color codes in hexadecimal format.
    """
    # take matplotlib standard color wheel
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # add extra color palette
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
    """Base class for all visualizers.

    Provides common functionality for visualization components including
    plot styling, color management, and file saving capabilities.

    Attributes:
        visualization_config (VisualizationConfig): Configuration for plots.
        output_config (OutputFolderConfig, optional): Output directory config.
        figsize (Tuple[float, float]): Default figure size for plots.
        colors (List[str]): Color palette for plots.
    """

    def __init__(
        self,
        visualization_config: VisualizationConfig,
        output_config: Optional[OutputFolderConfig] = None,
    ):
        """Initialize the base visualizer.

        Args:
            visualization_config (VisualizationConfig): Plot configuration settings.
            output_config (OutputFolderConfig, optional): Output folder configuration.
        """
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
        """Save the figure and show it."""
        if fig is None:
            fig = plt.gcf()

        if self.visualization_config.save_figures:
            if filename and self.output_config:
                if not filename.endswith(".png"):
                    filename = filename + ".png"
                full_path = os.path.join(self.output_path, filename)
                fig.savefig(full_path, bbox_inches="tight")

        if self.visualization_config.show_figures:
            plt.show(block=False)
            plt.pause(0.001)
            plt.show()
        else:
            plt.close(fig)
