"""Cox time-varying-covariate model — Feynman's ICU framing.

Treats each part as a patient moving through 51 stations (wards).
Estimates per-station hazard contributions to failure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from lifelines import CoxPHFitter

logger = logging.getLogger(__name__)


@dataclass
class CoxResult:
    fitted: CoxPHFitter
    concordance: float
    summary_df: pd.DataFrame


def prepare_cox_frame(
    features: pd.DataFrame,
    transit: pd.DataFrame,
    y: pd.Series,
    duration_col: str = "transit_time_total",
    event_col: str = "Response",
) -> pd.DataFrame:
    """Combine engineered features + transit-time + event flag into one frame."""
    frame = features.copy()
    frame[duration_col] = transit[duration_col].values
    frame[event_col] = y.values
    frame = frame.dropna(subset=[duration_col])
    frame = frame[frame[duration_col] > 0]
    return frame


def fit_cox(frame: pd.DataFrame, cfg: dict) -> CoxResult:
    surv_cfg = cfg["survival_cox"]
    cph = CoxPHFitter(penalizer=surv_cfg["penalizer"])
    logger.info("fitting Cox model on %d rows, %d covariates", len(frame), frame.shape[1] - 2)
    cph.fit(
        frame,
        duration_col=surv_cfg["duration_col"],
        event_col=surv_cfg["event_col"],
        show_progress=False,
        fit_options={"step_size": surv_cfg.get("step_size", 0.5)},
    )
    concordance = cph.concordance_index_
    logger.info("Cox concordance index = %.4f", concordance)
    return CoxResult(fitted=cph, concordance=concordance, summary_df=cph.summary)


def top_hazard_stations(result: CoxResult, top_k: int = 15) -> pd.DataFrame:
    df = result.summary_df.copy()
    df["abs_coef"] = df["coef"].abs()
    return df.sort_values("abs_coef", ascending=False).head(top_k)
