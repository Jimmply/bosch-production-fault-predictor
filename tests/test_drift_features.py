"""Tests for the drift-aware feature helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drift_features import add_drift_features, rolling_zscore


def test_rolling_zscore_first_rows_return_zero_before_window_fills():
    s = pd.Series(np.arange(50, dtype=float))
    z = rolling_zscore(s, window=10, min_periods=10)
    # First 10 rows can't compute z (window hasn't filled given shift(1))
    assert (z.iloc[:10] == 0.0).all()
    # Later rows should be finite and non-zero
    assert z.iloc[20:].abs().mean() > 0


def test_rolling_zscore_uses_only_past_values():
    # Value at position i must be z-scored against strictly-earlier rows only
    s = pd.Series([0.0] * 20 + [10.0] + [0.0] * 20)
    z = rolling_zscore(s, window=5, min_periods=5)
    # The spike at position 20 should be z-scored against 20 zeros -> zero-std -> filled 0
    # The row AFTER the spike (position 21) sees the spike in its window and returns non-zero
    assert z.iloc[20] == 0.0  # zero std in prior window
    assert z.iloc[21] != 0.0  # spike is now in the baseline


def test_rolling_zscore_no_lookahead():
    s = pd.Series([0.0] * 30 + [1000.0])
    z = rolling_zscore(s, window=10, min_periods=10)
    # The last row must not have been influenced by itself — its baseline is prior zeros
    # (zero std -> 0.0 output), so it must still be 0.0 exactly
    assert z.iloc[-1] == 0.0


def test_add_drift_features_appends_expected_columns():
    df = pd.DataFrame({
        "a": np.random.default_rng(0).normal(size=200),
        "b": np.random.default_rng(1).normal(size=200),
    })
    out = add_drift_features(df, columns=["a", "b"], window=20, min_periods=20)
    assert "a__z20" in out.columns
    assert "b__z20" in out.columns
    assert out.shape == (200, 4)
    # Original columns are unchanged
    pd.testing.assert_series_equal(out["a"], df["a"])


def test_add_drift_features_custom_suffix():
    df = pd.DataFrame({"x": np.linspace(0, 1, 100)})
    out = add_drift_features(df, ["x"], window=10, min_periods=10, suffix="drift")
    assert "x__drift" in out.columns


def test_add_drift_features_rejects_missing_column():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(KeyError):
        add_drift_features(df, ["a", "does_not_exist"], window=2, min_periods=1)
