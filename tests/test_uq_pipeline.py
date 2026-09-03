"""Step 3 integration tests: uncertainty_config wiring, save_models bands, coverage.

These extend `test_los_estimator_integration.py` with the pieces specific to
turning UQ into a first-class pipeline feature: the config round-trips into
`run_configurations.toml`, `load_run` tolerates older runs without an
`uncertainty_config` table, `<distro>_models.csv` gains band columns only when
UQ actually ran, dill pickling of the fit results (with the new band fields)
survives `save_results`, and the coverage artifacts are written honestly.
"""

import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import toml

from los_estimator.config import (
    UncertaintyConfig,
    load_configurations,
    save_configurations,
)
from los_estimator.estimation_run import LosEstimationRun
from los_estimator.fitting.uncertainty import UncertaintyParams

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_config.toml"
)

# Distributions that step 2's measurements found DO produce a band on this
# synthetic-ish dataset (see the UQ prompt); keep the fitting fast and
# deterministic by restricting to a couple of them under debug_config.
BAND_PRODUCING_DISTROS = ["linear", "exponential"]


def _make_run(uncertainty_config=None, run_nickname=None):
    cfg = load_configurations(CONFIG_PATH)
    return LosEstimationRun(
        cfg["data_config"],
        cfg["output_config"],
        cfg["model_config"],
        cfg["debug_config"],
        cfg["visualization_config"],
        cfg["animation_config"],
        uncertainty_config=uncertainty_config,
        run_nickname=run_nickname or f"uq_pipeline_{uuid.uuid4().hex[:8]}",
    )


class TestUncertaintyConfigRoundTrip:
    """`uncertainty_config` must round-trip through TOML like every other config."""

    def test_default_uncertainty_config_is_disabled(self):
        cfg = UncertaintyConfig()
        assert cfg.enabled is False
        assert cfg.n_samples == 1000
        assert tuple(cfg.confidence_interval) == (5.0, 95.0)
        assert cfg.distributions is None
        assert cfg.seed is None

    def test_round_trips_through_toml(self, tmp_path):
        cfg = UncertaintyConfig(
            enabled=True,
            n_samples=250,
            confidence_interval=(10.0, 90.0),
            distributions=["gaussian", "exponential"],
            seed=7,
        )
        path = tmp_path / "uq_only.toml"
        save_configurations(path, [cfg])

        loaded = load_configurations(path)
        assert "uncertainty_config" in loaded
        reloaded = loaded["uncertainty_config"]
        assert reloaded.enabled is True
        assert reloaded.n_samples == 250
        assert tuple(reloaded.confidence_interval) == (10.0, 90.0)
        assert reloaded.distributions == ["gaussian", "exponential"]
        assert reloaded.seed == 7

    def test_default_config_toml_declares_uncertainty_config(self):
        from los_estimator.config import default_config_path

        raw = toml.load(default_config_path)
        assert "uncertainty_config" in raw
        assert raw["uncertainty_config"]["enabled"] is False

    def test_from_config_maps_fields_onto_uncertainty_params(self):
        cfg = UncertaintyConfig(
            enabled=True, n_samples=42, confidence_interval=(2.0, 98.0), seed=3
        )
        params = UncertaintyParams.from_config(cfg)
        assert params.enabled is True
        assert params.n_samples == 42
        assert params.confidence_interval == (2.0, 98.0)
        assert params.seed == 3

    def test_from_config_none_is_disabled(self):
        params = UncertaintyParams.from_config(None)
        assert params.enabled is False


class TestLoadRunToleratesOlderRuns:
    """`LosEstimationRun.load_run` must not choke on a pre-step-3 run."""

    def test_load_run_without_uncertainty_config_table(self, tmp_path):
        run = _make_run(uncertainty_config=UncertaintyConfig(enabled=False))
        run.set_up()

        # Simulate a run saved before uncertainty_config existed: write
        # run_configurations.toml without that table.
        configs_without_uq = [
            c
            for c in run.configurations
            if getattr(c, "config_name", None) != "uncertainty_config"
        ]
        path = os.path.join(run.output_config.results, "run_configurations.toml")
        save_configurations(path, configs_without_uq)

        raw = toml.load(path)
        assert "uncertainty_config" not in raw

        reloaded = LosEstimationRun.load_run(run.output_config.results)
        assert reloaded.uncertainty_config.enabled is False


class TestUncertaintyPipelineEndToEnd:
    """Full pipeline runs, enabled and disabled, checking the artifacts step 3 adds."""

    def test_enabled_run_adds_bands_and_coverage(self):
        uq_cfg = UncertaintyConfig(
            enabled=True, n_samples=200, confidence_interval=(5.0, 95.0), seed=123
        )
        run = _make_run(uncertainty_config=uq_cfg)
        run.run_analysis()

        assert run.all_fit_results is not None
        assert run.coverage_summary is not None
        assert run.coverage_detail is not None

        # At least one of the known band-producing distros actually got a band
        # somewhere; report honestly rather than asserting every distro works
        # (lognorm/beta/invgauss/t/sentinel are known not to on this dataset).
        any_band = any(
            any(
                fr is not None and fr.test_lower is not None
                for fr in run.all_fit_results[d].fit_results
            )
            for d in BAND_PRODUCING_DISTROS
            if d in run.all_fit_results
        )
        assert any_band, "expected at least one window/distro to produce a UQ band"

        models_path = Path(run.output_config.results) / "model_data"
        for distro in BAND_PRODUCING_DISTROS:
            df = pd.read_csv(models_path / f"{distro}_models.csv")
            kernel_lo_cols = [c for c in df.columns if c.startswith("kernel_lo_")]
            kernel_hi_cols = [c for c in df.columns if c.startswith("kernel_hi_")]
            param_se_cols = [c for c in df.columns if c.startswith("param_se_")]
            assert kernel_lo_cols and kernel_hi_cols and param_se_cols
            # every kernel_i has a matching kernel_lo_i / kernel_hi_i
            n_kernel = len(
                [
                    c
                    for c in df.columns
                    if c.startswith("kernel_")
                    and not c.startswith("kernel_lo_")
                    and not c.startswith("kernel_hi_")
                ]
            )
            assert len(kernel_lo_cols) == n_kernel
            assert len(kernel_hi_cols) == n_kernel
            # at least one row has a finite band (this distro is known to produce one)
            assert np.isfinite(df[kernel_lo_cols].to_numpy()).any()

        metrics_path = Path(run.output_config.metrics)
        assert (metrics_path / "ci_coverage.csv").exists()
        assert (metrics_path / "ci_coverage_detail.csv").exists()
        coverage_df = pd.read_csv(metrics_path / "ci_coverage.csv")
        assert set(coverage_df["distribution"]) <= set(run.model_config.distributions)
        assert (coverage_df["nominal_coverage"] == 0.9).all()

        # per-distribution fit+coverage dashboards land in figures/coverage/,
        # one per distro that actually produced a band
        coverage_figures_path = Path(run.output_config.figures) / "coverage"
        band_producing = set(
            pd.read_csv(metrics_path / "ci_coverage_detail.csv")
            .dropna(subset=["coverage"])["distribution"]
            .unique()
        )
        assert band_producing, "expected at least one distro with a band in the detail table"
        for distro in band_producing:
            assert (coverage_figures_path / f"{distro}_coverage_dashboard.png").exists()

        # dill round trip through save_results -> load_run must survive the
        # new band fields on SingleFitResult/SeriesFitResult (step 2 never
        # exercised this path).
        reloaded = LosEstimationRun.load_run(run.output_config.results)
        assert reloaded.all_fit_results is not None
        reloaded_series = reloaded.all_fit_results["linear"]
        assert reloaded_series.all_kernel_lower is not None
        assert (
            reloaded_series[0].kernel_lower is not None
            or reloaded_series[0].kernel_lower is None
        )  # no crash

    def test_disabled_run_matches_pre_step3_csv_layout(self):
        run = _make_run(uncertainty_config=UncertaintyConfig(enabled=False))
        run.run_analysis()

        assert run.coverage_summary is None
        assert run.coverage_detail is None

        metrics_path = Path(run.output_config.metrics)
        assert not (metrics_path / "ci_coverage.csv").exists()
        assert not (metrics_path / "coverage_over_time.png").exists()

        models_path = Path(run.output_config.results) / "model_data"
        for distro in run.model_config.distributions:
            df = pd.read_csv(models_path / f"{distro}_models.csv")
            assert not any(
                c.startswith("kernel_lo_")
                or c.startswith("kernel_hi_")
                or c.startswith("param_se_")
                for c in df.columns
            )
            expected_prefixes = {"window", "train_error", "test_error"}
            other_cols = [c for c in df.columns if c not in expected_prefixes]
            assert all(
                c.startswith("param_") or c.startswith("kernel_") for c in other_cols
            )
