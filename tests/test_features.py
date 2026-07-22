"""Smoke tests for src/features.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import (
    group_columns_by_station,
    missing_pattern_features,
    parse_station,
    station_aggregates,
    transit_time_features,
)


def test_parse_station_recognizes_bosch_pattern() -> None:
    tag = parse_station("L0_S12_F45")
    assert tag is not None
    assert tag.line == 0 and tag.station == 12
    assert tag.group_key == "L0_S12"


def test_parse_station_returns_none_for_non_bosch_column() -> None:
    assert parse_station("Response") is None
    assert parse_station("Id") is None


def test_group_columns_by_station() -> None:
    cols = ["L0_S1_F1", "L0_S1_F2", "L0_S2_F1", "L1_S0_F1", "Response"]
    groups = group_columns_by_station(cols)
    assert set(groups["L0_S1"]) == {"L0_S1_F1", "L0_S1_F2"}
    assert groups["L0_S2"] == ["L0_S2_F1"]
    assert "Response" not in [c for cs in groups.values() for c in cs]


def _synthetic_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame(
        {
            "L0_S1_F1": rng.normal(size=n),
            "L0_S1_F2": rng.normal(size=n),
            "L0_S2_F1": rng.normal(size=n),
            "L1_S0_F1": rng.normal(size=n),
        }
    )


def test_station_aggregates_returns_expected_columns() -> None:
    df = _synthetic_df()
    agg = station_aggregates(df)
    for station in ["L0_S1", "L0_S2", "L1_S0"]:
        for suffix in ["mean", "std", "notna", "abs_sum"]:
            assert f"{station}__{suffix}" in agg.columns


def test_transit_time_features_positive() -> None:
    date_df = pd.DataFrame(
        {
            "L0_S1_D1": [10.0, 20.0, 30.0],
            "L0_S2_D1": [15.0, 25.0, 35.0],
            "L1_S0_D1": [12.0, 22.0, 32.0],
        }
    )
    transit = transit_time_features(date_df)
    assert (transit["transit_time_total"] >= 0).all()
    assert (transit["n_stations_visited"] == 3).all()


def test_missing_pattern_features_shape() -> None:
    df = _synthetic_df()
    df.loc[0:10, "L0_S1_F1"] = np.nan
    missing = missing_pattern_features(df)
    assert "L0__missing_frac" in missing.columns
    assert "L1__missing_frac" in missing.columns
    assert (missing["L0__missing_frac"] >= 0).all()
    assert (missing["L0__missing_frac"] <= 1).all()
