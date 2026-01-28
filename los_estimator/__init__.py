"""Length of Stay Estimator Package

A package for analyzing hospital length of stay using deconvolution methods.
"""

import logging
import sys
from pathlib import Path

__version__ = "1.0.0"


def setup_logging(log_file_path=None):
    """Configure logging for the whole package.

    Sets up a logger with both console and optional file output.

    Args:
        log_file_path (str, optional): Path to log file. If provided, logs will
            also be written to this file. Parent directories will be created
            if they don't exist.

    Returns:
        logging.Logger: Configured logger instance for the los_estimator package.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("los_estimator")

    if log_file_path:
        Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(fh)
    return logger


# Only setup logging when not being imported by Sphinx
if "sphinx" not in sys.modules:
    setup_logging("current.log")

# Configuration classes
from .config import (
    DataConfig,
    ModelConfig,
    VisualizationContext,
    DebugConfig,
    OutputFolderConfig,
    AnimationConfig,
    VisualizationConfig,
)

# Core classes
from .core import *

# Data loading
from .data import DataLoader
from .estimation_run import LosEstimationRun

# Fitting algorithms
from .fitting import MultiSeriesFitter

# Visualization components
from .visualization import (
    DeconvolutionAnimator,
    DeconvolutionPlots,
)

__all__ = [
    "LosEstimationRun",
    # Core classes
    "ModelConfig",
    "WindowInfo",
    "SeriesData",
    "SingleFitResult",
    "SeriesFitResult",
    "MultiSeriesFitResults",
    "Utils",
    # Configuration
    "ModelConfig",
    "DataConfig",
    "DebugConfig",
    "OutputFolderConfig",
    "AnimationConfig",
    "VisualizationConfig",
    "VisualizationContext",
    # Data
    "DataLoader",
    # Fitting
    "MultiSeriesFitter",
    # Visualization
    "DeconvolutionPlots",
    "DeconvolutionAnimator",
]
