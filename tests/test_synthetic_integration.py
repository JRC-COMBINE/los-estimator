"""Fast synthetic-data integration test.

Unlike `test_los_estimator_integration.py` and `test_uq_pipeline.py`, which load the
real packaged ICU dataset via `tests/test_config.toml`, this test fabricates its own
admissions/occupancy series with `generate_and_save_synthetic_data` and runs the same
`LosEstimationRun.run_analysis()` pipeline against it. It exists to give a smoke test
that runs in seconds on every local iteration, with no dependency on real data.
"""

import sys
import uuid
from pathlib import Path

import numpy as np

sys.path.append(Path(__file__).parents[1].as_posix())
sys.path.append((Path(__file__).parents[1] / "examples").as_posix())

from generate_synthetic_data import generate_and_save_synthetic_data
from los_estimator.config import (
    AnimationConfig,
    DataConfig,
    DebugConfig,
    ModelConfig,
    OutputFolderConfig,
    VisualizationConfig,
)
from los_estimator.estimation_run import LosEstimationRun

# `generate_and_save_synthetic_data` convolves admissions with a lognorm kernel
# built from these [sigma, mu] parameters (see examples/generate_synthetic_data.py);
# fitting "lognorm" back out should recover values close to these.
TRUE_LOGNORM_PARAMS = np.array([1.2, 0.7 + np.log(10)])
REL_LOGNORM_PARAM_TOLERANCE = 1 / 100


def test_synthetic_estimation_run_completes_successfully(tmp_path):
    """Runs the full pipeline on generated data with a minimal, fast config."""
    kernel_width, data_path, kernel_path = generate_and_save_synthetic_data(length=300, output_dir=str(tmp_path))

    data_config = DataConfig(
        icu_file=data_path,
        los_file=kernel_path,
        start_day="2020-01-01",
        end_day="2021-02-03",
    )
    # `base` feeds directly into a long, auto-generated results folder name (config
    # values baked into the path) - keep this as short as possible to stay under
    # Windows' ~260 char path limit once nested output files are added.
    output_config = OutputFolderConfig(base=str(tmp_path), run_name="")
    model_config = ModelConfig(
        kernel_width=kernel_width,
        train_width=110,
        test_width=14,
        step=7,
        distributions=["linear", "exponential", "lognorm"],
    )
    debug_config = DebugConfig(one_window=True)
    visualization_config = VisualizationConfig(
        show_figures=False,
        save_figures=False,
        colors=["#1f77b4", "#ff7f0e", "#2ca02c"],
    )
    animation_config = AnimationConfig(show_figures=False, save_figures=False)

    short_id = uuid.uuid4().hex[:8]
    estimator = LosEstimationRun(
        data_config,
        output_config,
        model_config,
        debug_config,
        visualization_config,
        animation_config,
        run_nickname=f"synth_{short_id}",
    )
    estimator.run_analysis()

    assert estimator.all_fit_results is not None
    assert len(estimator.all_fit_results) > 0

    for distro in model_config.distributions:
        assert distro in estimator.all_fit_results
        series_result = estimator.all_fit_results[distro]
        assert any(fr is not None for fr in series_result.fit_results)

    # Check the recovered lognorm kernel parameters, not just that fitting ran:
    # at least one window's fitted [sigma, mu] should land close to the values
    # used to generate the synthetic occupancy series (TRUE_LOGNORM_PARAMS).
    lognorm_fits = [fr for fr in estimator.all_fit_results["lognorm"].fit_results if fr is not None]
    assert lognorm_fits
    closest_error = min(
        np.max(np.abs(fr.distro_params - TRUE_LOGNORM_PARAMS) / TRUE_LOGNORM_PARAMS) for fr in lognorm_fits
    )
    assert closest_error < REL_LOGNORM_PARAM_TOLERANCE, (
        f"no fitted window recovered lognorm params within "
        f"{REL_LOGNORM_PARAM_TOLERANCE:.2%} of {TRUE_LOGNORM_PARAMS} "
        f"(closest max-relative-error: {closest_error})"
    )
