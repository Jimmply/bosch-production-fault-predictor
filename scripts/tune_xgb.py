"""Optuna hyperparameter search for the baseline XGBoost.

Runs a small TPE search over max_depth, learning_rate, min_child_weight,
subsample, colsample_bytree. Uses a stratified sample by default for speed;
best params are written to config/tuned_params.yaml for train.py to pick up.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import optuna
import yaml
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data_loader import load_config  # noqa: E402
from scripts_utils_bridge import load_engineered_sample  # noqa: E402  # thin wrapper around train.load_train_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def objective(trial: optuna.Trial, X, y, cfg: dict) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "scale_pos_weight": cfg["baseline_xgb"]["scale_pos_weight"],
        "eval_metric": "aucpr",
        "random_state": 42,
        "n_jobs": -1,
    }
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    mccs = []
    for tr, va in kf.split(X, y):
        model = XGBClassifier(**params)
        model.fit(X.iloc[tr], y.iloc[tr], verbose=False)
        proba = model.predict_proba(X.iloc[va])[:, 1]
        # Sweep a coarse threshold grid to find fold-optimal MCC
        best = 0.0
        for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            m = matthews_corrcoef(y.iloc[va], (proba >= t).astype(int))
            if m > best:
                best = m
        mccs.append(best)
    return sum(mccs) / len(mccs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--sample-n", type=int, default=200_000)
    args = parser.parse_args()

    cfg = load_config()
    logger.info("loading engineered sample (n=%d) for tuning...", args.sample_n)
    X, y = load_engineered_sample(cfg, sample_n=args.sample_n)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda t: objective(t, X, y, cfg), n_trials=args.n_trials, show_progress_bar=False)

    best = study.best_params
    logger.info("best MCC = %.4f", study.best_value)
    logger.info("best params = %s", best)

    out = REPO_ROOT / "config" / "tuned_params.yaml"
    with open(out, "w") as f:
        yaml.safe_dump({"best_mcc": float(study.best_value), "params": best}, f)
    logger.info("saved -> %s", out)


if __name__ == "__main__":
    main()
