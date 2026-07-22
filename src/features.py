"""Feature engineering: station-group aggregates, transit times, missing-pattern features."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STATION_PATTERN = re.compile(r"^L(\d+)_S(\d+)_")


@dataclass
class StationTag:
    line: int
    station: int

    @property
    def group_key(self) -> str:
        return f"L{self.line}_S{self.station}"


def parse_station(col: str) -> StationTag | None:
    match = STATION_PATTERN.match(col)
    if not match:
        return None
    return StationTag(line=int(match.group(1)), station=int(match.group(2)))


def group_columns_by_station(columns: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for col in columns:
        tag = parse_station(col)
        if tag is None:
            continue
        groups.setdefault(tag.group_key, []).append(col)
    return groups


def station_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Per-station aggregates: mean, std, count of non-null, sum-of-abs."""
    groups = group_columns_by_station(df.columns.tolist())
    out = {}
    for station, cols in groups.items():
        sub = df[cols]
        out[f"{station}__mean"] = sub.mean(axis=1)
        out[f"{station}__std"] = sub.std(axis=1)
        out[f"{station}__notna"] = sub.notna().sum(axis=1)
        out[f"{station}__abs_sum"] = sub.abs().sum(axis=1)
    return pd.DataFrame(out, index=df.index)


def transit_time_features(date_df: pd.DataFrame) -> pd.DataFrame:
    """First-timestamp / last-timestamp / total transit per part from date_df."""
    ts_cols = [c for c in date_df.columns if c.startswith("L")]
    first_ts = date_df[ts_cols].min(axis=1)
    last_ts = date_df[ts_cols].max(axis=1)
    return pd.DataFrame(
        {
            "transit_time_first": first_ts,
            "transit_time_last": last_ts,
            "transit_time_total": last_ts - first_ts,
            "n_stations_visited": date_df[ts_cols].notna().sum(axis=1),
        },
        index=date_df.index,
    )


def missing_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-line missingness fraction — often predictive on Bosch."""
    groups = group_columns_by_station(df.columns.tolist())
    lines: dict[int, list[str]] = {}
    for station, cols in groups.items():
        line = int(station.split("_")[0][1:])
        lines.setdefault(line, []).extend(cols)
    return pd.DataFrame(
        {f"L{line}__missing_frac": df[cols].isna().mean(axis=1) for line, cols in lines.items()},
        index=df.index,
    )
