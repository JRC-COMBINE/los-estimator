# %%
# %load_ext autoreload
# %autoreload 2
from los_estimator.estimation_run import LosEstimationRun, load_configurations, load_configurations
from los_estimator.config import update_configurations, default_config_path

print("Let's Go!")

# %%
cfg = load_configurations(default_config_path)
overwrite_cfg = load_configurations(default_config_path.parent / "overwrite_config.toml")


model_config = cfg["model_config"]
data_config = cfg["data_config"]
output_config = cfg["output_config"]
debug_config = cfg["debug_config"]
visualization_config = cfg["visualization_config"]
animation_config = cfg["animation_config"]
uncertainty_config = cfg["uncertainty_config"]


# USe the following line to apply configuration overrides
# update_configurations(cfg, overwrite_cfg)

# The configurations can also be modified directly here
debug_config.less_windows = False
debug_config.less_distros = False


model_config.distributions = [
    "lognorm",
    "weibull",
    "gaussian",
    "exponential",
    "gamma",
    "beta",
    "cauchy",
    "t",
    "invgauss",
    "linear",
    "sentinel",
    "compartmental",
]


# %%
estimator = LosEstimationRun(
    data_config, output_config, model_config, debug_config, visualization_config, animation_config, uncertainty_config
)
estimator.run_analysis()
# %%
# Use the following lines to step through the analysis manually
# self = estimator
# self.set_up()
# self.load_data()
# self.fit()
# self.evaluate()
# self.save_results()
# self.visualize_metrics()
# self.visualize_results()
# self.animate_results()
# %%
