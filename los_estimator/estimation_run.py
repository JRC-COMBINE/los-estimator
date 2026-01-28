import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

import dill

from los_estimator.config import (
    DataConfig,
    ModelConfig,
    VisualizationContext,
    DebugConfig,
    OutputFolderConfig,
    AnimationConfig,
    VisualizationConfig,
    load_configurations,
    save_configurations,
)
from los_estimator.core import (
    SeriesData,
)
from los_estimator.data import DataLoader, DataPackage
from los_estimator.evaluation import Evaluator
from los_estimator.fitting import MultiSeriesFitter
from los_estimator.fitting.fit_results import MultiSeriesFitResults
from los_estimator.visualization import (
    DeconvolutionAnimator,
    DeconvolutionPlots,
    get_color_palette,
)
from los_estimator.visualization.metrics import MetricsPlots

logger = logging.getLogger("los_estimator")


class LosEstimationRun:
    """Main class for running Length of Stay estimation analysis.

    This class orchestrates the entire LOS estimation process including data loading,
    model fitting, evaluation, and visualization. It manages configurations and
    coordinates between different components of the analysis pipeline.

    Attributes:
        configurations (List): List of all configuration objects.
        run_nickname (Optional[str]): Optional nickname for this analysis run.
        model_config (ModelConfig): Configuration for model parameters.
        output_config (OutputFolderConfig): Configuration for output handling.
        data_config (DataConfig): Configuration for data loading and processing.
        debug_config (DebugConfig): Configuration for debugging options.
    """

    @staticmethod
    def load_run(folder):
        path = Path(folder)

        cfg = load_configurations(path / "run_configurations.toml")
        model_config = cfg["model_config"]
        data_config = cfg["data_config"]
        output_config = cfg["output_config"]
        debug_config = cfg["debug_config"]
        visualization_config = cfg["visualization_config"]
        animation_config = cfg["animation_config"]

        run_nickname = None
        if "run_nickname" in cfg:
            run_nickname = cfg["run_nickname"]

        run = LosEstimationRun(
            data_config,
            output_config,
            model_config,
            debug_config,
            visualization_config,
            animation_config,
            run_nickname,
        )

        def _load(name):
            file = path / "model_data" / f"{name}.pkl"
            if file.exists():
                with open(file, "rb") as f:
                    return dill.load(f)
            return None

        run.series_data = _load("series_data")
        run.all_fit_results = _load("all_fit_results")
        run.visualization_context = _load("visualization_context")

        return run

    def __init__(
        self,
        data_config: DataConfig,
        output_config: OutputFolderConfig,
        model_config: ModelConfig,
        debug_config: DebugConfig,
        visualization_config: VisualizationConfig,
        animation_config: AnimationConfig,
        run_nickname: Optional[str] = None,
    ):
        """Initialize LOS estimation run with configurations.

        Args:
            data_config (DataConfig): Configuration for data loading and processing.
            output_config (OutputFolderConfig): Configuration for output folder structure.
            model_config (ModelConfig): Configuration for model parameters and fitting.
            debug_config (DebugConfig): Configuration for debugging and development options.
            visualization_config (VisualizationConfig): Configuration for plot generation.
            animation_config (AnimationConfig): Configuration for animation generation.
            run_nickname (Optional[str], optional): Nickname for this run. Defaults to None.
        """
        self.configurations = [
            data_config,
            output_config,
            model_config,
            debug_config,
            visualization_config,
            animation_config,
        ]
        self.run_nickname = run_nickname
        self.model_config: ModelConfig = model_config
        self.output_config: OutputFolderConfig = output_config
        self.data_config: DataConfig = data_config
        self.debug_config: DebugConfig = debug_config
        self.visualization_config: VisualizationConfig = visualization_config
        self.animation_config: AnimationConfig = animation_config

        self.visualization_context: VisualizationContext = VisualizationContext()
        self.data: DataPackage = None

        self.create_run()
        output_config.run_name = self.run_name
        output_config.build()

        if self.visualization_config.colors is None:
            self.visualization_config.colors = get_color_palette()

        self.data_loader: DataLoader = DataLoader(data_config)

        self.fitter: MultiSeriesFitter = None
        self.window_data: list = None
        self.all_fit_results: MultiSeriesFitResults = None
        self.series_data: SeriesData = None
        self.evaluator: Evaluator = None
        self.data_loaded = False

    def load_data(self):
        """Load all required data for the analysis.

        Loads data using the configured DataLoader and sets up visualization context
        with the loaded data properties like x-axis labels and real LOS values.
        """
        self.data = self.data_loader.load_all_data()

        vc = self.visualization_context
        vc.xtick_pos = self.data.xtick_pos
        vc.xtick_label = self.data.xtick_label
        vc.real_los = self.data.real_los

        vc.xlims = self.visualization_config.xlims
        vc.results_folder = self.output_config.results
        vc.figures_folder = self.output_config.figures
        vc.animation_folder = self.output_config.animation

        self.data_loaded = True

    def set_up(self):
        """Set up the output directory structure and logging.

        Creates necessary output directories, removes existing results if present,
        and configures logging for the analysis run.
        """
        c = self.output_config

        c.run_name = self.run_name
        c.build()

        Path(c.results).mkdir(parents=True, exist_ok=True)
        Path(c.figures).mkdir(parents=True, exist_ok=True)
        Path(c.animation).mkdir(parents=True, exist_ok=True)
        Path(c.metrics).mkdir(parents=True, exist_ok=True)

        self.set_up_logger()

        if self.debug_config.less_distros:
            self.model_config.distributions = ["linear", "exponential"]
        if self.debug_config.only_linear:
            self.model_config.distributions = ["linear"]

    def set_up_logger(self):
        """Configure file logging for this analysis run.

        Adds a file handler to the logger to capture all log messages
        in a run-specific log file within the results directory.
        """
        path = Path(self.output_config.results) / "run.log"
        file_handler = logging.FileHandler(path)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

    def visualize_metrics(self):
        """Generate visualizations of evaluation metrics.

        Creates plots for evaluation metrics using the MetricsPlots visualizer.
        """
        metrics_plots = MetricsPlots(
            series_data=self.series_data,
            visualization_config=self.visualization_config,
            visualization_context=self.visualization_context,
            output_config=self.output_config,
            evaluation_results=self.evaluator.result,
        )
        metrics_plots.plot_metrics()

    def visualize_results(self):
        """Generate visualizations of the fitting results.

        Creates deconvolution plots and other result visualizations if visualization
        is enabled in the configuration. Skips visualization if both show_figures
        and save_figures are disabled.
        """
        if not self.visualization_config.show_figures and not self.visualization_config.save_figures:
            logger.info("Visualization is disabled. Skipping visualization.")
            return
        self.deconv_plot_visualizer = DeconvolutionPlots(
            self.all_fit_results,
            self.series_data,
            self.model_config,
            self.visualization_config,
            self.visualization_context,
            self.output_config,
        )
        self.deconv_plot_visualizer.generate_plots_for_run()

    def animate_results(self):
        if not self.animation_config.show_figures and not self.animation_config.save_figures:
            logger.info("Animation is disabled. Skipping animation creation.")
            return
        self.animator = DeconvolutionAnimator(
            all_fit_results=self.all_fit_results,
            series_data=self.series_data,
            model_config=self.model_config,
            visualization_config=self.visualization_config,
            visualization_context=self.visualization_context,
            animation_config=self.animation_config,
            output_folder_config=self.output_config,
        )

        self.animator.animate_fit_deconvolution()

        if self.animation_config.generate_gif:
            self.animator.combine_to_gif()

    def create_run(self):
        model_config = self.model_config
        if model_config.run_name is not None and model_config.run_name != "":
            self.run_name = model_config.run_name
            return
        timestamp = time.strftime("%y%m%d_%H%M")

        run_name = f"{timestamp}_dev"

        run_name += f"_step{model_config.step}_train{model_config.train_width}_test{model_config.test_width}"
        run_name += "_fit_admissions"
        run_name += "_" + model_config.error_fun
        if model_config.reuse_last_parametrization:
            run_name += "_reuse_last_parametrization"
        if model_config.iterative_kernel_fit:
            run_name += "_iterative_kernel_fit"

        run_name += f"_{self.run_nickname}" if self.run_nickname else ""
        model_config.run_name = run_name
        self.run_name = run_name

    def run_analysis(self):

        self.set_up()
        self.load_data()

        self.fit()

        self.evaluate()
        self.save_results()

        self.visualize_metrics()

        self.visualize_results()

        self.animate_results()

        logger.info(f"Results saved in: {self.output_config.results}")
        logger.info("LOS estimation run completed.")

    def fit(self):

        series_data = (
            self.data.df_occupancy["icu_admissions"].values,
            self.data.df_occupancy["icu_occupancy"].values,
        )
        self.series_data = SeriesData(*series_data, self.model_config, self.debug_config)

        init_parameters = defaultdict(list)
        if self.data.df_init is not None:
            for distro, row in self.data.df_init.iterrows():
                init_parameters[distro] = row["params"]

        self.fitter = MultiSeriesFitter(
            self.series_data,
            self.model_config,
            self.model_config.distributions,
            init_parameters,
        )
        self.fitter.DEBUG_MODE(self.debug_config)

        self.window_data, self.all_fit_results = self.fitter.fit()
        return self.window_data, self.all_fit_results

    def evaluate(self):
        if self.all_fit_results is None:
            raise ValueError("No fit results available. Please run the fit method first.")
        self.evaluator = Evaluator(
            all_fit_results=self.all_fit_results,
            series_data=self.series_data,
        )
        self.evaluator.calculate_metrics()

    def save_results(self):
        path = os.path.join(self.output_config.results, "run_configurations.toml")
        save_configurations(path, self.configurations)

        to_save = {}
        if self.series_data is not None:
            to_save["series_data"] = self.series_data
        if self.all_fit_results is not None:
            to_save["all_fit_results"] = self.all_fit_results
        if self.visualization_context is not None:
            to_save["visualization_context"] = self.visualization_context

        Path(self.output_config.model_data).mkdir(parents=True, exist_ok=True)
        for name, data in to_save.items():
            path = os.path.join(self.output_config.model_data, f"{name}.pkl")
            with open(path, "wb") as f:
                dill.dump(data, f)
        if self.evaluator is not None:
            self.evaluator.save_result(self.output_config.metrics)
        self.save_models()

    def save_models(self):
        models_path = Path(self.output_config.results) / "model_data"
        models_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving fitted models to {models_path.as_posix()}")

        for distro in self.model_config.distributions:

            series_fit_result = self.all_fit_results[distro]
            window_ids = [w.window for w in series_fit_result.window_infos]
            distro_params = np.array([fr.distro_params for fr in series_fit_result])
            kernels = series_fit_result.all_kernels[window_ids]
            train_errors = series_fit_result.train_errors
            test_errors = series_fit_result.test_errors
            len_config = distro_params.shape[1]
            len_kernels = kernels.shape[1]
            data_dict = {
                "window": window_ids,
                "train_error": train_errors,
                "test_error": test_errors,
            }
            data_dict.update({f"param_{i}": distro_params[:, i] for i in range(len_config)})
            data_dict.update({f"kernel_{i}": kernels[:, i] for i in range(len_kernels)})
            df = pd.DataFrame(data_dict)
            df.to_csv(models_path / f"{distro}_models.csv", index=False)

    logger.info("Model saving complete.")
