"""Tests for los_estimator.data.DataLoader.

test_load_los_normalizes_read_only_array is a regression test: pandas
copy-on-write can make Series.to_numpy() return a read-only array, and
normalizing it in place (`los /= los.sum()`) used to raise
`ValueError: assignment destination is read-only`.
"""

import numpy as np
import pandas as pd

from los_estimator.data import DataLoader


def test_load_los_normalizes_read_only_array(tmp_path, monkeypatch):
    csv_path = tmp_path / "los.csv"
    pd.DataFrame({"los": [1.0, 2.0, 3.0, 4.0]}).to_csv(csv_path)

    original_to_numpy = pd.Series.to_numpy

    def read_only_to_numpy(self, *args, **kwargs):
        arr = original_to_numpy(self, *args, **kwargs).copy()
        arr.setflags(write=False)
        return arr

    monkeypatch.setattr(pd.Series, "to_numpy", read_only_to_numpy)

    loader = DataLoader(data_config=None)
    los = loader.load_los(str(csv_path))

    assert np.isclose(los.sum(), 1.0)


def test_load_los_returns_none_when_no_file():
    loader = DataLoader(data_config=None)
    assert loader.load_los(None) is None
