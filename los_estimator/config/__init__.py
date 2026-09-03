"""Model configuration for LOS Estimator."""

import os
import types
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import List, Optional, Tuple

from pyparsing import Union
import toml

__all__ = [
    "ModelConfig",
    "DataConfig",
    "DebugConfig",
    "OutputFolderConfig",
    "AnimationConfig",
    "VisualizationConfig",
    "VisualizationContext",
    "UncertaintyConfig",
    "load_configurations",
    "save_configurations",
    "default_config_path",
    "update_configurations",
]

default_config_path = Path(__file__).parent.parent / "default_config.toml"

configuration_type = {}


def config(name=None):
    """Turn a class into a dataclass and register it under `name` (or its class name)
    in `configuration_type`, keyed by the TOML table name it loads from."""

    def wrapper(cls):
        cls = dataclass(cls)
        _name = name if name else cls.__name__
        cls.config_name = _name
        configuration_type[_name] = cls
        return cls

    return wrapper


@config("model_config")
class ModelConfig:
    """Fitting/windowing parameters. `train_width` must exceed `kernel_width`
    to avoid left-edge truncation."""

    kernel_width: int = 120
    train_width: int = 42 + 60
    test_width: int = 21
    step: int = 7
    error_fun: str = "mse"
    reuse_last_parametrization: bool = True
    iterative_kernel_fit: bool = True
    distributions: List[str] = field(
        default_factory=lambda: [
            "lognorm",
            # "weibull",
            "gaussian",
            "exponential",
            # "gamma",
            # "beta",
            "cauchy",
            "t",
            # "invgauss",
            "linear",
            # "sentinel",
            "compartmental",
        ]
    )
    optimizer: str = "L-BFGS-B"
    run_name: str = ""


@config("data_config")
class DataConfig:
    """File paths and date range for the input data. Paths may use the `${data}`
    placeholder, resolved to the packaged `los_estimator/data/` dir."""

    icu_file: str
    los_file: Optional[str] = None
    start_day: Optional[str] = None
    end_day: Optional[str] = None

    init_params_file: Optional[str] = None

    def __post_init__(self):
        """No-op; reserved for future validation."""


@config("debug_config")
class DebugConfig:
    """Flags to shrink the run for fast iteration (fewer windows/distributions)."""

    one_window: bool = False
    less_windows: bool = False
    less_distros: bool = False
    only_linear: bool = False


@config("output_config")
class OutputFolderConfig:
    """Derives the results/figures/animation/metrics/model_data subfolder
    paths from `base`/`run_name`."""

    base: str
    run_name: str

    def build(self):
        if not self.run_name:
            return
        self.results = os.path.join(self.base, self.run_name)
        self.figures = os.path.join(self.results, "figures")
        self.animation = os.path.join(self.results, "animation")
        self.metrics = os.path.join(self.results, "metrics")
        self.model_data = os.path.join(self.results, "model_data")

    def __post_init__(self):
        self.build()


@config("animation_config")
class AnimationConfig:
    """Controls for the fit-progression GIF/animation output."""

    show_figures: bool = False
    save_figures: bool = True
    generate_gif: bool = True
    short_distro_names: List[Tuple[str, str]] = field(
        default_factory=lambda: [
            ("exponential", "exp"),
            ("gaussian", "gauss"),
            ("compartmental", "comp"),
        ]
    )
    train_error_lim: Union[str, float] = "auto"
    test_error_lim: Union[str, float] = "auto"


@config("visualization_config")
class VisualizationConfig:
    """Plot generation controls.

    file_format: extension (no dot) that `VisualizerBase._show` saves as, e.g.
    "png" or "pdf" — passed straight to `Figure.savefig`, so anything
    Matplotlib supports works.
    """

    save_figures: bool = True
    show_figures: bool = True

    xlims: Tuple[int, int] = (-30, 725)
    figsize: Tuple[int, int] = (12, 8)
    style: str = "seaborn-v0_8"
    colors: List[str] = field(default_factory=lambda: [])
    savefig_facecolor = "white"
    savefig_dpi: int = 300
    figure_dpi: int = 100
    file_format: str = "png"


@config("uncertainty_config")
class UncertaintyConfig:
    """Drives the Laplace-approximation uncertainty pass
    (`los_estimator.fitting.uncertainty.UncertaintyParams.from_config`).
    Off by default. `distributions=None` means all supported distributions."""

    enabled: bool = False
    n_samples: int = 1000
    confidence_interval: Tuple[float, float] = (5.0, 95.0)
    distributions: Optional[List[str]] = None
    seed: Optional[int] = None


@dataclass
class VisualizationContext:
    """Shared axis formatting and output paths passed to all visualizers."""

    xtick_pos: Tuple = ()
    xtick_label: Tuple = ()
    real_los: Tuple = ()
    xlims: Tuple = (-30, 725)
    results_folder: str = ""
    figures_folder: str = ""
    animation_folder: str = ""


def dict_to_config(config_dict, config_class):
    """Instantiate config_class from config_dict, dropping unknown keys."""
    field_names = {field.name for field in fields(config_class)}
    filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
    return config_class(**filtered_dict)


def load_configurations(path):
    """Load a TOML file into {table_name: config_object}, skipping unregistered tables."""
    with open(path, "r") as f:
        loaded_config = toml.load(f)

    configs = {}
    for name in loaded_config.keys():

        if name not in configuration_type:
            continue

        configs[name] = dict_to_config(loaded_config[name], configuration_type[name])
    return configs


def save_configurations(path, configurations):
    """Write a list of config objects to a TOML file, keyed by their config_name."""
    config_dicts = {config.config_name: asdict(config) for config in configurations}
    with open(path, "w") as f:
        toml.dump(config_dicts, f)


def update_configurations(base_config, override_config):
    """Recursively merge override_config into base_config in place."""
    for key, value in override_config.items():
        if isinstance(value, dict):
            update_configurations(base_config.setdefault(key, {}), value)
        else:
            base_config[key] = value
