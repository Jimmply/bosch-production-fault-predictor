"""Drift-aware features — rolling z-scored versions of raw feature columns.

Hypothesis (unverified as of this module landing): the time-aware AUC collapse
comes from the physical process changing across the training-set time span
(see docs/img/drift_analysis.png). Windowed retraining did NOT fix it
(see docs/img/windowed_experiment.png). The next thing to try is feature-level
adaptation: instead of feeding the model the raw sensor value, feed it the
sensor's *deviation from the recent baseline*.

Given a time-sorted DataFrame and a set of feature columns, `add_drift_features`
returns those columns z-scored against a rolling window that ends STRICTLY
BEFORE each row — no lookahead. Uses shift(1) so a row is never in its own
baseline window.

Not yet wired into train.py — this is the scaffold for the next experiment.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def rolling_zscore(series: pd.Series, window: int, min_periods: int = 100) -> pd.Series:
    """Rolling z-score using ONLY past values (shift(1) means row i is not in its own baseline).

    Rows before the rolling window has filled up return 0.0 (neutral). Windows
    with zero std also return 0.0 to avoid inf.
    """
    past = series.shift(1).rolling(window=window, min_periods=min_periods)
    mean = past.mean()
    std = past.std().replace(0.0, np.nan)
    z = (series - mean) / std
    return z.fillna(0.0)


def add_drift_features(
    df: pd.DataFrame,
    columns: list[str],
    window: int = 5000,
    min_periods: int = 200,
    suffix: str | None = None,
) -> pd.DataFrame:
    """Return a copy of `df` with rolling-z-scored versions of `columns` appended.

    New column names follow ``<col>__z<window>`` unless `suffix` is passed.

    The caller is responsible for ordering `df` by time first — this function
    treats the row order as the temporal order.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"columns not in dataframe: {missing}")

    tag = suffix if suffix is not None else f"z{window}"
    new_cols = {}
    for col in columns:
        new_cols[f"{col}__{tag}"] = rolling_zscore(df[col], window=window, min_periods=min_periods)
    out = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    logger.info("added %d drift features from %d source columns (window=%d)",
                len(new_cols), len(columns), window)
    return out
