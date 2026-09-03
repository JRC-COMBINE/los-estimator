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
    UncertaintyConfig,
    load_configurations,
    save_configurations,
)
from los_estimator.core import (
    SeriesData,
)
from los_estimator.data import DataLoader, DataPackage
from los_estimator.evaluation import Evaluator
from los_estimator.evaluation.coverage import (
    compute_coverage_table,
    save_coverage_tables,
    summarize_coverage,
)
from los_estimator.fitting import MultiSeriesFitter
from los_estimator.fitting.distributions import Distributions
from los_estimator.fitting.fit_results import MultiSeriesFitResults
from los_estimator.fitting.uncertainty import UncertaintyParams
from los_estimator.visualization import (
    CoveragePlots,
    DeconvolutionAnimator,
    DeconvolutionPlots,
    get_color_palette,
)
from los_estimator.visualization.metrics import MetricsPlots

logger = logging.getLogger("los_estimator")


class LosEstimationRun:
    """Orchestrates the full pipeline: data loading, fitting, evaluation, and
    visualization, driven by `run_analysis()`."""

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
        # Older runs were saved before uncertainty_config existed; tolerate the
        # missing table and fall back to disabled defaults.
        uncertainty_config = cfg.get("uncertainty_config", UncertaintyConfig())

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
            uncertainty_config=uncertainty_config,
            run_nickname=run_nickname,
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
        uncertainty_config: Optional[UncertaintyConfig] = None,
        run_nickname: Optional[str] = None,
    ):
        if uncertainty_config is None:
            uncertainty_config = UncertaintyConfig()

        self.configurations = [
            data_config,
            output_config,
            model_config,
            debug_config,
            visualization_config,
            animation_config,
            uncertainty_config,
        ]
        self.run_nickname = run_nickname
        self.model_config: ModelConfig = model_config
        self.output_config: OutputFolderConfig = output_config
        self.data_config: DataConfig = data_config
        self.debug_config: DebugConfig = debug_config
        self.visualization_config: VisualizationConfig = visualization_config
        self.animation_config: AnimationConfig = animation_config
        self.uncertainty_config: UncertaintyConfig = uncertainty_config
        self.coverage_detail = None
        self.coverage_summary = None

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
        path = Path(self.output_config.results) / "run.log"
        file_handler = logging.FileHandler(path)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

    def visualize_metrics(self):
        metrics_plots = MetricsPlots(
            series_data=self.series_data,
            visualization_config=self.visualization_config,
            visualization_context=self.visualization_context,
            output_config=self.output_config,
            evaluation_results=self.evaluator.result,
        )
        metrics_plots.plot_metrics()

    def visualize_results(self):
        if (
            not self.visualization_config.show_figures
            and not self.visualization_config.save_figures
        ):
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

        if self.coverage_detail is not None:
            self.deconv_plot_visualizer.show_coverage_dashboards(self.coverage_detail)

    def animate_results(self):
        if (
            not self.animation_config.show_figures
            and not self.animation_config.save_figures
        ):
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
        self.validate_coverage()
        self.save_results()

        self.visualize_metrics()
        self.visualize_coverage()

        self.visualize_results()

        self.animate_results()

        logger.info(f"Results saved in: {self.output_config.results}")
        logger.info("LOS estimation run completed.")

    def fit(self):

        series_data = (
            self.data.df_occupancy["icu_admissions"].values,
            self.data.df_occupancy["icu_occupancy"].values,
        )
        self.series_data = SeriesData(
            *series_data, self.model_config, self.debug_config
        )

        init_parameters = defaultdict(list)
        if self.data.df_init is not None:
            for distro, row in self.data.df_init.iterrows():
                params = row["params"]
                try:
                    expected = Distributions.n_parameters(distro)
                except ValueError:
                    logger.warning(
                        f"Ignoring initial parameters for unknown distribution '{distro}'."
                    )
                    continue
                if len(params) != expected:
                    # Legacy files carry a trailing stretch factor for every
                    # distribution; only the ones in USES_SCALING still have one.
                    logger.warning(
                        f"Ignoring initial parameters for '{distro}': got {len(params)} values, "
                        f"expected {expected}. Falling back to the built-in defaults."
                    )
                    continue
                init_parameters[distro] = params

        self.fitter = MultiSeriesFitter(
            self.series_data,
            self.model_config,
            self.model_config.distributions,
            init_parameters,
            uncertainty=UncertaintyParams.from_config(self.uncertainty_config),
        )
        self.fitter.DEBUG_MODE(self.debug_config)

        self.window_data, self.all_fit_results = self.fitter.fit()
        return self.window_data, self.all_fit_results

    def evaluate(self):
        if self.all_fit_results is None:
            raise ValueError(
                "No fit results available. Please run the fit method first."
            )
        self.evaluator = Evaluator(
            all_fit_results=self.all_fit_results,
            series_data=self.series_data,
        )
        self.evaluator.calculate_metrics()

    def validate_coverage(self):
        """Compute empirical coverage of the UQ test bands, if UQ is enabled.

        No-op (leaves `coverage_detail`/`coverage_summary` at None) when
        `uncertainty_config.enabled` is False, so a disabled run does not pay
        for or emit coverage artifacts.
        """
        if not self.uncertainty_config.enabled:
            return
        if self.all_fit_results is None:
            raise ValueError(
                "No fit results available. Please run the fit method first."
            )
        self.coverage_detail = compute_coverage_table(
            self.all_fit_results,
            self.model_config.kernel_width,
            self.uncertainty_config.confidence_interval,
        )
        self.coverage_summary = summarize_coverage(self.coverage_detail)
        for _, row in self.coverage_summary.iterrows():
            logger.info(
                f"UQ coverage [{row['distribution']}]: "
                f"mean={row['mean_coverage']:.3f} "
                f"effective={row['mean_coverage_effective']:.3f} "
                f"vs nominal={row['nominal_coverage']:.3f} "
                f"({row['n_windows_with_band']}/{row['n_windows_total']} windows with a band, "
                f"band_rate={row['band_rate']:.2f})"
            )

    def visualize_coverage(self):
        """Render the coverage-over-time figure into `metrics/`, if available."""
        if self.coverage_detail is None:
            return
        if (
            not self.visualization_config.show_figures
            and not self.visualization_config.save_figures
        ):
            return
        CoveragePlots(
            self.coverage_detail,
            self.visualization_config,
            self.output_config,
        ).plot_coverage_over_time()

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
        if self.coverage_summary is not None and self.coverage_detail is not None:
            save_coverage_tables(
                self.coverage_detail, self.coverage_summary, self.output_config.metrics
            )
        self.save_models()

    def save_models(self):
        models_path = Path(self.output_config.results) / "model_data"
        models_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving fitted models to {models_path.as_posix()}")

        # Only append UQ columns when the pass actually ran, so a disabled run
        # produces the exact same CSV layout as before this feature existed.
        add_uq_columns = self.uncertainty_config.enabled

        for distro in self.model_config.distributions:

            series_fit_result = self.all_fit_results[distro]
            window_ids = [w.window for w in series_fit_result.window_infos]
            distro_params = np.array([fr.distro_params for fr in series_fit_result])
            kernels = series_fit_result.all_kernels[window_ids]
            train_errors = series_fit_result.train_errors
            test_errors = series_fit_result.test_errors
            len_config = distro_params.shape[1]
            len_kernels = kernels.shape[1]
            n_windows = len(series_fit_result.fit_results)
            data_dict = {
                "window": window_ids,
                "train_error": train_errors,
                "test_error": test_errors,
            }
            data_dict.update(
                {f"param_{i}": distro_params[:, i] for i in range(len_config)}
            )
            data_dict.update({f"kernel_{i}": kernels[:, i] for i in range(len_kernels)})

            if add_uq_columns:
                param_se = np.full((n_windows, len_config), np.nan)
                for i, fr in enumerate(series_fit_result.fit_results):
                    if fr is not None and fr.covariance is not None:
                        diag = np.diag(
                            np.atleast_2d(np.asarray(fr.covariance, dtype=float))
                        )
                        m = min(len(diag), len_config)
                        param_se[i, :m] = np.sqrt(np.clip(diag[:m], 0, None))
                data_dict.update(
                    {f"param_se_{i}": param_se[:, i] for i in range(len_config)}
                )

                kernel_lower = series_fit_result.all_kernel_lower
                kernel_upper = series_fit_result.all_kernel_upper
                if kernel_lower is None or kernel_lower.shape != (
                    n_windows,
                    len_kernels,
                ):
                    kernel_lower = np.full((n_windows, len_kernels), np.nan)
                if kernel_upper is None or kernel_upper.shape != (
                    n_windows,
                    len_kernels,
                ):
                    kernel_upper = np.full((n_windows, len_kernels), np.nan)
                data_dict.update(
                    {f"kernel_lo_{i}": kernel_lower[:, i] for i in range(len_kernels)}
                )
                data_dict.update(
                    {f"kernel_hi_{i}": kernel_upper[:, i] for i in range(len_kernels)}
                )

            df = pd.DataFrame(data_dict)
            df.to_csv(models_path / f"{distro}_models.csv", index=False)

        logger.info("Model saving complete.")
