"""Fit Cox proportional-hazards model on top-K SHAP-attributed stations.

Feynman's ICU framing: treat each part as a patient moving through 51 wards,
estimate per-station hazard contributions to quality-control failure.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data_loader import load_config, resolve_paths
from features import missing_pattern_features, station_aggregates, transit_time_features
from survival_model import fit_cox, prepare_cox_frame, top_hazard_stations

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main(top_k_stations: int = 30, sample_n: int = 300_000) -> None:
    cfg = load_config()
    paths = resolve_paths(cfg)
    id_col = cfg["loading"]["id_col"]
    target_col = cfg["loading"]["target_col"]

    logger.info("loading train_numeric.parquet")
    numeric = pd.read_parquet(paths.parquet_dir / "train_numeric.parquet")

    if len(numeric) > sample_n:
        pos = numeric[numeric[target_col] == 1]
        neg = numeric[numeric[target_col] == 0].sample(n=sample_n - len(pos), random_state=cfg["split"]["random_seed"])
        numeric = pd.concat([pos, neg], ignore_index=False).sample(frac=1.0, random_state=cfg["split"]["random_seed"])
    numeric = numeric.reset_index(drop=True)

    logger.info("loading train_date.parquet")
    date_df = pd.read_parquet(paths.parquet_dir / "train_date.parquet")
    date_df = date_df[date_df[id_col].isin(set(numeric[id_col].values))].copy()
    merged_date = numeric[[id_col]].merge(date_df, on=id_col, how="left").reset_index(drop=True).drop(columns=[id_col])
    transit = transit_time_features(merged_date)

    y = numeric[target_col].astype(int).reset_index(drop=True)
    features = numeric.drop(columns=[target_col, id_col])
    agg = station_aggregates(features)
    miss = missing_pattern_features(features)

    logger.info("loading SHAP attribution to pick top-%d stations", top_k_stations)
    attribution = pd.read_parquet(REPO_ROOT / "models" / "station_attribution.parquet")
    top_stations = attribution.head(top_k_stations)["station"].tolist()
    logger.info("top stations: %s", top_stations[:10])

    keep_cols = [c for c in agg.columns if any(c.startswith(s + "__") for s in top_stations)]
    logger.info("selected %d station-aggregate columns for Cox", len(keep_cols))

    selected = agg[keep_cols].fillna(0.0)
    # Drop constant columns which break Cox partial-likelihood optimization
    variances = selected.var(axis=0)
    keep_variant = variances[variances > 1e-10].index.tolist()
    dropped = set(keep_cols) - set(keep_variant)
    if dropped:
        logger.info("dropping %d constant columns: %s", len(dropped), sorted(dropped)[:5])
    selected = selected[keep_variant]

    frame = prepare_cox_frame(selected, transit, y)
    logger.info("Cox frame: %d rows, %d covariates, %d events", len(frame), frame.shape[1] - 2, frame[target_col].sum())

    result = fit_cox(frame, cfg)
    top = top_hazard_stations(result, top_k=15)
    logger.info("top hazard-driving covariates:\n%s", top[["coef", "exp(coef)", "p"]].to_string())

    out_dir = REPO_ROOT / "models"
    out_dir.mkdir(exist_ok=True)
    result.summary_df.to_csv(out_dir / "cox_summary.csv")
    with open(out_dir / "cox_concordance.txt", "w") as f:
        f.write(f"{result.concordance:.6f}\n")
    logger.info("saved Cox summary + concordance = %.4f", result.concordance)


if __name__ == "__main__":
    main()
