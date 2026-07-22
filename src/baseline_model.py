"""Baseline XGBoost classifier with proper MCC scoring and imbalance handling."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


@dataclass
class BaselineMetrics:
    mcc_folds: list[float]
    auc_folds: list[float]
    aucpr_folds: list[float]

    @property
    def summary(self) -> dict[str, float]:
        return {
            "mcc_mean": float(np.mean(self.mcc_folds)),
            "mcc_std": float(np.std(self.mcc_folds)),
            "auc_mean": float(np.mean(self.auc_folds)),
            "aucpr_mean": float(np.mean(self.aucpr_folds)),
        }


def optimal_threshold_by_mcc(y_true: np.ndarray, y_proba: np.ndarray, n_thresh: int = 200) -> tuple[float, float]:
    """Sweep thresholds and return (best_threshold, best_mcc)."""
    thresholds = np.linspace(0.01, 0.99, n_thresh)
    best_mcc, best_t = -1.0, 0.5
    for t in thresholds:
        mcc = matthews_corrcoef(y_true, (y_proba >= t).astype(int))
        if mcc > best_mcc:
            best_mcc, best_t = mcc, t
    return float(best_t), float(best_mcc)


def _make_splitter(split_cfg: dict):
    strategy = split_cfg.get("strategy", "stratified")
    if strategy == "time":
        return TimeSeriesSplit(n_splits=split_cfg["n_folds"])
    return StratifiedKFold(n_splits=split_cfg["n_folds"], shuffle=True, random_state=split_cfg["random_seed"])


def train_baseline(X: pd.DataFrame, y: pd.Series, cfg: dict) -> tuple[XGBClassifier, dict]:
    params = cfg["baseline_xgb"]
    split_cfg = cfg["split"]
    kf = _make_splitter(split_cfg)
    logger.info("split strategy: %s (n_folds=%d)", split_cfg.get("strategy", "stratified"), split_cfg["n_folds"])

    mcc_folds, auc_folds, aucpr_folds = [], [], []
    split_iter = kf.split(X, y) if split_cfg.get("strategy", "stratified") != "time" else kf.split(X)
    for fold, (train_idx, val_idx) in enumerate(split_iter):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBClassifier(**{k: v for k, v in params.items() if k != "early_stopping_rounds"})
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        proba = model.predict_proba(X_val)[:, 1]

        _, best_mcc = optimal_threshold_by_mcc(y_val.values, proba)
        auc = roc_auc_score(y_val, proba)
        aucpr = average_precision_score(y_val, proba)
        mcc_folds.append(best_mcc)
        auc_folds.append(auc)
        aucpr_folds.append(aucpr)
        logger.info("fold %d: MCC=%.4f AUC=%.4f AUCPR=%.4f", fold, best_mcc, auc, aucpr)

    metrics = BaselineMetrics(mcc_folds, auc_folds, aucpr_folds).summary

    final_model = XGBClassifier(**{k: v for k, v in params.items() if k != "early_stopping_rounds"})
    final_model.fit(X, y, verbose=False)
    return final_model, metrics
