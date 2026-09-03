"""Core data classes and structures for LOS estimation."""

import functools
from typing import Optional

import numpy as np

from los_estimator.config import DebugConfig, ModelConfig

__all__ = [
    "WindowInfo",
    "SeriesData",
]


class WindowInfo:
    """Indices/slices for one sliding-analysis window; `window` is the train/test boundary index."""

    def __init__(self, window: int, model_config: ModelConfig):
        self.window: int = window
        self.kernel_width: int = model_config.kernel_width

        self.train_end: int = self.window
        self.train_start: int = self.train_end - model_config.train_width

        self.test_start: int = self.train_end
        self.test_end: int = self.test_start + model_config.test_width
        self.training_prediction_start: int = self.train_start + model_config.kernel_width

        self.train_window: slice = slice(self.train_start, self.train_end)
        self.train_test_window: slice = slice(self.train_start, self.test_end)
        self.test_window: slice = slice(self.test_start, self.test_end)
        self.test_window_with_runup: slice = slice(self.test_start - model_config.kernel_width, self.test_end)

        self.model_config: ModelConfig = model_config

    def __repr__(self):
        return f"WindowInfo(window={self.window}, train_start={self.train_start}, train_end={self.train_end}, test_start={self.test_start}, test_end={self.test_end})"


class SeriesData:
    """Admissions/occupancy series plus the precomputed sliding windows over it."""

    def __init__(
        self,
        x_full: np.ndarray,
        y_full: np.ndarray,
        model_config: ModelConfig,
        debug_config: Optional[DebugConfig] = None,
    ):
        self.model_config: ModelConfig = model_config
        self.x_full: np.ndarray = x_full
        self.y_full: np.ndarray = y_full
        self.windows: np.ndarray
        self.window_infos: list[WindowInfo]
        self.n_windows: int
        self.debug_config: DebugConfig = debug_config or DebugConfig()
        self._calc_windows(model_config)

        self.n_days: int = len(self.x_full)

    def _calc_windows(self, model_config):
        start = model_config.train_width
        windows = np.arange(start, len(self.x_full) - model_config.kernel_width, model_config.step)
        if self.debug_config.less_windows:
            windows = windows[:3]
        elif self.debug_config.one_window:
            windows = windows[10:11]
        self.windows = windows

        self.window_infos = [WindowInfo(window, model_config) for window in self.windows]
        self.n_windows = len(self.windows)

    @functools.lru_cache
    def get_train_data(self, window_id: int):
        if window_id > len(self.windows):
            raise ValueError(f"Window ID {window_id} out of range for {len(self.windows)} windows.")
        w = self.window_infos[window_id]
        return self.x_full[w.train_window], self.y_full[w.train_window]

    @functools.lru_cache
    def get_test_data(self, window_id: int):
        if window_id > len(self.windows):
            raise ValueError(f"Window ID {window_id} out of range for {len(self.windows)} windows.")
        w = self.window_infos[window_id]
        return (
            self.x_full[w.test_window_with_runup],
            self.y_full[w.test_window_with_runup],
        )

    @functools.lru_cache
    def get_window_info(self, window_id: int):
        if window_id > len(self.windows):
            raise ValueError(f"Window ID {window_id} out of range for {len(self.windows)} windows.")
        return self.window_infos[window_id]

    def __iter__(self):
        for idx in range(self.n_windows):
            train_data = self.get_train_data(idx)
            test_data = self.get_test_data(idx)
            window_info = self.get_window_info(idx)
            yield idx, window_info, train_data, test_data

    def __len__(self):
        return self.n_windows

    def __repr__(self):
        return f"SeriesData(n_windows={self.n_windows}, kernel_width={self.model_config.kernel_width}"
