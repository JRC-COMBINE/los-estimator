import os
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

sys.path.append(Path(__file__).parents[1].as_posix())

from los_estimator.config import default_config_path
from los_estimator.estimation_run import LosEstimationRun, load_configurations


class TestLosEstimatorIntegration:
    """Integration tests for LOS Estimator."""

    def test_estimation_run_completes_successfully(self):
        self.cfg = load_configurations(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_config.toml"))

        unique_id = str(uuid.uuid4())
        estimator = LosEstimationRun(
            self.cfg["data_config"],
            self.cfg["output_config"],
            self.cfg["model_config"],
            self.cfg["debug_config"],
            self.cfg["visualization_config"],
            self.cfg["animation_config"],
            run_nickname=f"test_run_{unique_id}",
        )
        estimator.run_analysis()

        # Basic assertions
        assert hasattr(estimator, "all_fit_results")
        assert estimator.all_fit_results is not None
        assert len(estimator.all_fit_results) > 0

    def test_cli_execution_completes_successfully(self):
        cli_script = Path(__file__).parents[1] / "los_estimator/__main__.py"

        if not cli_script.exists():
            pytest.skip("CLI script not found, skipping CLI test")

        cmd = [
            sys.executable,
            str(cli_script),
            "--config_file",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_config.toml"),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                check=True,
            )
            print("CLI stdout:", result.stdout)
            print("CLI stderr:", result.stderr)
            assert result.returncode == 0, f"CLI execution failed with return code {result.returncode}"

        except subprocess.TimeoutExpired:
            pytest.fail("CLI execution timed out after 5 minutes")
        except subprocess.CalledProcessError as e:
            pytest.fail(f"CLI execution failed: {e.stderr}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
