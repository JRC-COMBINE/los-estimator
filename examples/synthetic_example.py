# %%
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cwd = os.getcwd()

if not (cwd.endswith("examples") or cwd.endswith("examples/")):
    os.chdir("examples")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from generate_synthetic_data import generate_and_save_synthetic_data
from los_estimator.estimation_run import LosEstimationRun, load_configurations


# %%

kernel_width, data_path, kernel_path = generate_and_save_synthetic_data()

cfg = load_configurations("./synthetic_example.toml")

model_config = cfg["model_config"]
data_config = cfg["data_config"]
output_config = cfg["output_config"]
debug_config = cfg["debug_config"]
visualization_config = cfg["visualization_config"]
animation_config = cfg["animation_config"]

visualization_config.show_figures = False
animation_config.show_figures = False
data_config.icu_file = data_path
data_config.los_file = kernel_path
model_config.kernel_width = kernel_width

model_config.distributions = ["exponential", "linear", "gaussian"]
# debug_config.less_distros = True
# debug_config.less_windows = True

estimator = LosEstimationRun(
    data_config,
    output_config,
    model_config,
    debug_config,
    visualization_config,
    animation_config,
)
estimator.run_analysis()
# %%
