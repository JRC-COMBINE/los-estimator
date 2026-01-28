"""Data loading and preparation utilities for LOS estimation."""

from dataclasses import dataclass
from importlib import resources

import numpy as np
import pandas as pd

from ..config import DataConfig

__all__ = [
    "DataPackage",
    "DataUtils",
    "DataLoader",
]


@dataclass
class DataPackage:
    """Container for all loaded data required for LOS estimation.

    Attributes:
        df_occupancy (pd.DataFrame): ICU occupancy data over time.
        real_los (pd.Series): Real length of stay values for validation.
        df_init (pd.DataFrame): Initial condition data.
        xtick_pos (list): Positions for x-axis tick marks in plots.
        xtick_label (list): Labels for x-axis tick marks in plots.
    """

    df_occupancy: pd.DataFrame
    real_los: pd.Series
    df_init: pd.DataFrame
    xtick_pos: list
    xtick_label: list


class DataUtils:
    """Utility functions for data processing and manipulation.

    Provides static methods for common data operations like date conversions
    and generating axis labels for time series plots.
    """

    def generate_xticks(df):
        """Generate x-axis tick positions and labels for time series plots.

        Creates tick marks at the first day of each month, with year labels
        added for January or the first data point.

        Args:
            df (pd.DataFrame): DataFrame with datetime index.

        Returns:
            tuple: (xtick_pos, xtick_label) lists for plot formatting.
        """
        xtick_pos = []
        xtick_label = []
        for i in range(0, len(df)):
            if df.index[i].day == 1:  # first day of month
                xtick_pos.append(i)
                label = df.index[i].strftime("%b")
                if i == 0 or df.index[i].month == 1:
                    label += f"\n{df.index[i].year}"
                xtick_label.append(label)
        return xtick_pos, xtick_label


class DataLoader:
    """Data loader for LOS estimation datasets."""

    def __init__(self, data_config: DataConfig):
        self.data_config: DataConfig = data_config
        self.resource_path = str(resources.files("los_estimator").joinpath("data"))

    def _get_packaged_file_path(self, path):
        """Get path to packaged data file using importlib.resources."""
        if "${data}" in path:
            path = path.replace("${data}", self.resource_path)
        return path

    def read_csv(self, path, *args, **kwargs):
        """Read CSV file from packaged data."""
        file_path = self._get_packaged_file_path(path)
        return pd.read_csv(file_path, *args, **kwargs)

    def read_excel(self, path, *args, **kwargs) -> pd.DataFrame:
        """Read Excel file from packaged data."""
        file_path = self._get_packaged_file_path(path)
        return pd.read_excel(file_path, *args, **kwargs)

    def load_all_data(self):
        c = self.data_config

        df_occupancy = self.load_icu_data(c.start_day, c.end_day)

        # optional data
        df_init = self.load_init_parameters(c.init_params_file)
        real_los = self.load_los(c.los_file)

        xtick_pos = None
        xtick_label = None
        xtick_pos, xtick_label = DataUtils.generate_xticks(df_occupancy)

        return DataPackage(
            df_occupancy=df_occupancy,
            real_los=real_los,
            df_init=df_init,
            xtick_pos=xtick_pos,
            xtick_label=xtick_label,
        )

    def load_icu_data(self, start_day, end_day) -> pd.DataFrame:
        """Load incidence and ICU occupancy data."""
        c = self.data_config
        df_icu = self.read_csv(c.icu_file, parse_dates=["Unnamed: 0"])
        df_icu = df_icu.set_index("Unnamed: 0")

        df_icu = df_icu[df_icu.index >= start_day]
        df_icu = df_icu[df_icu.index <= end_day]

        return df_icu

    def load_init_parameters(self, file) -> pd.DataFrame:
        """Load initial parameters for the model."""
        if file is None:
            return None
        df_init = self.read_csv(file, index_col=0)
        df_init = df_init.set_index("distro")
        # interpret model_config as array float of format [f1 f2 f3 ...]
        df_init["params"] = df_init["params"].apply(lambda x: [float(i) for i in x[1:-1].split()])
        return df_init

    def load_los(self, file=None) -> np.ndarray:
        if file is None:
            return None
        df_los = self.read_csv(file, index_col=0)
        los = df_los.iloc[:, 0].to_numpy(dtype=float)
        los /= los.sum()
        return los
