"""End-to-end training pipeline: features -> baseline model -> SHAP attribution -> artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from baseline_model import optimal_threshold_by_mcc, train_baseline
from data_loader import load_config, resolve_paths
from features import missing_pattern_features, station_aggregates, transit_time_features
from station_attribution import aggregate_by_station, compute_shap_values, pareto_stations

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_train_frame(cfg: dict, sample_n: int | None) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    paths = resolve_paths(cfg)
    numeric_pq = paths.parquet_dir / "train_numeric.parquet"
    date_pq = paths.parquet_dir / "train_date.parquet"
    if not numeric_pq.exists():
        raise FileNotFoundError(f"missing {numeric_pq}; run scripts/download_data.py first")

    id_col = cfg["loading"]["id_col"]
    target_col = cfg["loading"]["target_col"]

    logger.info("loading %s", numeric_pq)
    numeric = pd.read_parquet(numeric_pq)
    logger.info("numeric shape %s | positive rate = %.4f", numeric.shape, numeric[target_col].mean())

    if sample_n is not None and len(numeric) > sample_n:
        pos = numeric[numeric[target_col] == 1]
        neg_needed = max(sample_n - len(pos), 0)
        neg = numeric[numeric[target_col] == 0].sample(n=neg_needed, random_state=cfg["split"]["random_seed"])
        numeric = pd.concat([pos, neg], ignore_index=False).sample(frac=1.0, random_state=cfg["split"]["random_seed"])
        logger.info("stratified sample -> %d rows (kept all %d positives)", len(numeric), len(pos))

    kept_ids = set(numeric[id_col].values)
    numeric = numeric.reset_index(drop=True)
    y = numeric[target_col].astype(int)
    numeric_features = numeric.drop(columns=[target_col, id_col])

    date_features = pd.DataFrame(index=numeric_features.index)
    if date_pq.exists():
        logger.info("loading %s and joining on Id", date_pq)
        date_df = pd.read_parquet(date_pq)
        date_df = date_df[date_df[id_col].isin(kept_ids)].copy()
        merged = numeric[[id_col]].merge(date_df, on=id_col, how="left")
        merged = merged.reset_index(drop=True).drop(columns=[id_col])
        date_features = transit_time_features(merged)
        date_features.index = numeric_features.index

    logger.info("engineering station aggregates + missing-pattern features...")
    agg = station_aggregates(numeric_features)
    miss = missing_pattern_features(numeric_features)

    X = pd.concat([agg, miss, date_features], axis=1).fillna(0)

    # If we're doing time-aware CV, X and y must be sorted by transit time so
    # TimeSeriesSplit walks forward in chronological order.
    if cfg["split"].get("strategy") == "time" and "transit_time_first" in X.columns:
        order = X["transit_time_first"].sort_values(kind="mergesort").index
        X = X.loc[order].reset_index(drop=True)
        y = y.loc[order].reset_index(drop=True)
        date_features = date_features.loc[order].reset_index(drop=True)
        logger.info("sorted by transit_time_first for time-aware CV")

    logger.info("engineered feature matrix: %s", X.shape)
    return X, y, date_features


def _maybe_apply_tuned_params(cfg: dict) -> None:
    """If config/tuned_params.yaml exists (written by scripts/tune_xgb.py), overlay its params on baseline_xgb."""
    tuned_path = REPO_ROOT / "config" / "tuned_params.yaml"
    if not tuned_path.exists():
        return
    with open(tuned_path) as f:
        tuned = yaml.safe_load(f) or {}
    params = tuned.get("params", {})
    if not params:
        logger.info("tuned_params.yaml exists but has no 'params' key — skipping")
        return
    overrides = {k: v for k, v in params.items() if k in cfg["baseline_xgb"] or k in {
        "n_estimators", "max_depth", "learning_rate", "min_child_weight",
        "subsample", "colsample_bytree",
    }}
    if not overrides:
        return
    logger.info("applying tuned params from %s: %s (tuned MCC = %s)",
                tuned_path.name, overrides, tuned.get("best_mcc", "?"))
    cfg["baseline_xgb"].update(overrides)


def save_artifacts(model, metrics: dict, threshold: float, attribution: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_dir / "baseline_xgb.json"))
    with open(out_dir / "metrics.yaml", "w") as f:
        yaml.safe_dump(metrics, f)
    with open(out_dir / "threshold.json", "w") as f:
        json.dump({"optimal_threshold_mcc": threshold}, f, indent=2)
    attribution.to_parquet(out_dir / "station_attribution.parquet", index=False)
    logger.info("saved artifacts to %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-n", type=int, default=None,
                        help="Stratified sample size (keeps all positives). Default: use full data.")
    parser.add_argument("--shap-max-samples", type=int, default=10_000,
                        help="Max rows for SHAP computation (SHAP on 1M+ rows OOMs).")
    parser.add_argument("--split-strategy", choices=["stratified", "time"], default=None,
                        help="Override split.strategy in config (stratified or time-aware CV).")
    args = parser.parse_args()

    cfg = load_config()
    _maybe_apply_tuned_params(cfg)
    if args.split_strategy is not None:
        cfg["split"]["strategy"] = args.split_strategy
        logger.info("overriding split.strategy = %s", args.split_strategy)
    X, y, transit = load_train_frame(cfg, sample_n=args.sample_n)
    logger.info("positive rate = %.4f (%d / %d)", y.mean(), y.sum(), len(y))

    model, metrics = train_baseline(X, y, cfg)
    logger.info("cv metrics: %s", metrics)

    proba = model.predict_proba(X)[:, 1]
    best_t, best_mcc = optimal_threshold_by_mcc(y.values, proba)
    logger.info("full-fit optimal threshold %.4f -> MCC %.4f", best_t, best_mcc)
    metrics["train_mcc_at_optimal_threshold"] = best_mcc

    logger.info("computing SHAP attribution on up to %d rows...", args.shap_max_samples)
    shap_df = compute_shap_values(model, X, max_samples=args.shap_max_samples)
    attribution = aggregate_by_station(shap_df)
    pareto = pareto_stations(attribution, cumulative_share=0.70)
    logger.info("Pareto: %d stations explain >=70%% of |SHAP|:\n%s", len(pareto), pareto.to_string(index=False))

    save_artifacts(model, metrics, best_t, attribution, REPO_ROOT / "models")


if __name__ == "__main__":
    main()
