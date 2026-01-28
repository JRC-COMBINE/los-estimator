import logging

import matplotlib.pyplot as plt

from ..config import (
    OutputFolderConfig,
    VisualizationConfig,
    VisualizationContext,
)
from ..core import SeriesData
from ..evaluation import EvaluationResult
from ..visualization.base import VisualizerBase

logger = logging.getLogger("los_estimator")


class MetricsPlots(VisualizerBase):

    def __init__(
        self,
        series_data: SeriesData,
        visualization_config: VisualizationConfig,
        visualization_context: VisualizationContext,
        output_config: OutputFolderConfig,
        evaluation_results: EvaluationResult,
    ):
        super().__init__(visualization_config, output_config)
        self.vc: VisualizationContext = visualization_context
        self.series_data: SeriesData = series_data
        self.evaluation_results: EvaluationResult = evaluation_results

        self.output_path = output_config.metrics

    def plot_metrics(self):
        logger.info("Plotting evaluation metrics...")
        eval_res = self.evaluation_results
        colors = self.visualization_config.colors

        for i_metric, metric in enumerate(eval_res.metric_names):
            self._figure(figsize=(12, 5))

            for i_distro, distro in enumerate(eval_res.distros):
                test_res = eval_res.test[i_distro, :, i_metric]
                plt.plot(
                    self.series_data.windows,
                    test_res,
                    label=f"{distro} - Test - {metric}",
                    color=colors[i_distro],
                )
            plt.title(f"{metric.capitalize()} Test Metric for Rolling Models")
            plt.xlabel("Model Training Date")
            plt.ylabel(f"Error ({metric})")
            xlim = plt.gca().get_xlim()
            plt.gca().set_xticks(self.vc.xtick_pos[1:-2:2])
            plt.gca().set_xticklabels(self.vc.xtick_label[1:-2:2])
            plt.xlim(xlim)
            plt.legend()
            self._show(f"{metric}_test.png")
