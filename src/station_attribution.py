"""Aggregate SHAP values by production station — Newton's angle.

Uses XGBoost's native `pred_contribs=True` to avoid the shap-library version incompat
with XGBoost 3.x, and because it's faster than the shap library on tree models.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xgboost as xgb

from features import group_columns_by_station

logger = logging.getLogger(__name__)


def compute_shap_values(model, X: pd.DataFrame, max_samples: int = 10_000) -> pd.DataFrame:
    """Compute SHAP contributions using XGBoost's native output. Drops the bias column."""
    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=42)
        logger.info("subsampled X to %d rows for SHAP", max_samples)
    dmat = xgb.DMatrix(X)
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    contribs = booster.predict(dmat, pred_contribs=True)
    # pred_contribs returns shape (n, n_features + 1); last col is the bias term.
    return pd.DataFrame(contribs[:, :-1], columns=X.columns, index=X.index)


def aggregate_by_station(shap_df: pd.DataFrame) -> pd.DataFrame:
    """Return per-station aggregate |SHAP| contribution and share of total."""
    groups = group_columns_by_station(shap_df.columns.tolist())
    per_station_abs_sum = {station: shap_df[cols].abs().sum().sum() for station, cols in groups.items()}
    total = sum(per_station_abs_sum.values())
    rows = [
        {
            "station": station,
            "line": int(station.split("_")[0][1:]),
            "station_num": int(station.split("_")[1][1:]),
            "abs_shap_sum": val,
            "share_of_total": val / total if total > 0 else np.nan,
        }
        for station, val in per_station_abs_sum.items()
    ]
    return pd.DataFrame(rows).sort_values("share_of_total", ascending=False).reset_index(drop=True)


def pareto_stations(attribution: pd.DataFrame, cumulative_share: float = 0.70) -> pd.DataFrame:
    """Return the smallest set of stations that cumulatively explain `cumulative_share` of |SHAP|."""
    ordered = attribution.sort_values("share_of_total", ascending=False).reset_index(drop=True)
    ordered["cumulative_share"] = ordered["share_of_total"].cumsum()
    cutoff_idx = ordered["cumulative_share"].searchsorted(cumulative_share) + 1
    return ordered.head(cutoff_idx)
