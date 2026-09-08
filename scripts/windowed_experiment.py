"""Windowed / online-training experiment.

Question: given that time-aware CV showed AUC collapse (0.717 -> 0.563) and
drift analysis found the physical process moving across the training-set time
span (~4x swing in defect rate), does retraining on RECENT data actually beat
the single global fit?

Approach: sort all 1.18M parts by transit_time_first, split into K=5 equal-count
sequential blocks. For each eval block b (b = 1..K-1):

  - windowed(b)         : train on the immediately preceding block only
                          (a "just the last window" recency-only model)
  - cumulative(b)       : train on ALL blocks strictly before b
                          (a "keep growing training set" model)
  - global-baseline(b)  : load the shipped tuned model (trained on everything)
                          and evaluate on block b

We record AUC and MCC for each (strategy, eval_block) pair. Cumulative uses
tuned hyperparameters from config/tuned_params.yaml; windowed uses the same.

Outputs:
  - models/windowed_metrics.csv           (numbers for the README table)
  - docs/img/windowed_experiment.png      (chart with 3 curves)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from data_loader import load_config, resolve_paths
from features import missing_pattern_features, station_aggregates, transit_time_features
from train import _maybe_apply_tuned_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def build_dataset(cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    paths = resolve_paths(cfg)
    id_col = cfg["loading"]["id_col"]
    target_col = cfg["loading"]["target_col"]

    logger.info("loading train_numeric.parquet")
    numeric = pd.read_parquet(paths.parquet_dir / "train_numeric.parquet")
    logger.info("loading train_date.parquet")
    date_df = pd.read_parquet(paths.parquet_dir / "train_date.parquet")

    date_df = date_df.set_index(id_col).loc[numeric[id_col].values].reset_index()
    transit = transit_time_features(date_df.drop(columns=[id_col]))

    y = numeric[target_col].astype(int).reset_index(drop=True)
    features = numeric.drop(columns=[target_col, id_col])
    agg = station_aggregates(features)
    miss = missing_pattern_features(features)
    X = pd.concat([agg, miss, transit.reset_index(drop=True)], axis=1).fillna(0)

    # Sort by first-station timestamp — this is the chronological order
    order = X["transit_time_first"].sort_values(kind="mergesort").index
    X = X.loc[order].reset_index(drop=True)
    y = y.loc[order].reset_index(drop=True)
    logger.info("dataset ready: %s, positive rate=%.4f", X.shape, y.mean())
    return X, y


def make_blocks(n: int, k: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, n, k + 1, dtype=int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(k)]


def fit_and_score(X_tr, y_tr, X_te, y_te, params: dict) -> dict:
    model = XGBClassifier(**params)
    model.fit(X_tr, y_tr, verbose=False)
    proba = model.predict_proba(X_te)[:, 1]
    yhat = (proba >= 0.5).astype(int)
    # threshold-optimal MCC on the eval block (small grid)
    best_mcc = -1.0
    for t in np.linspace(0.05, 0.95, 19):
        m = matthews_corrcoef(y_te, (proba >= t).astype(int))
        if m > best_mcc:
            best_mcc = m
    return {
        "auc": float(roc_auc_score(y_te, proba)),
        "mcc_at_0_5": float(matthews_corrcoef(y_te, yhat)),
        "mcc_at_optimal": float(best_mcc),
        "n_train": int(len(X_tr)),
        "n_eval": int(len(X_te)),
    }


def score_global_baseline(X_te, y_te) -> dict:
    """Load the shipped tuned XGBoost, score on the eval block."""
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(str(REPO_ROOT / "models" / "baseline_xgb.json"))
    proba = model.predict_proba(X_te)[:, 1]
    yhat = (proba >= 0.5).astype(int)
    best_mcc = -1.0
    for t in np.linspace(0.05, 0.95, 19):
        m = matthews_corrcoef(y_te, (proba >= t).astype(int))
        if m > best_mcc:
            best_mcc = m
    return {
        "auc": float(roc_auc_score(y_te, proba)),
        "mcc_at_0_5": float(matthews_corrcoef(y_te, yhat)),
        "mcc_at_optimal": float(best_mcc),
        "n_train": -1,
        "n_eval": int(len(X_te)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-blocks", type=int, default=5)
    args = parser.parse_args()
    k = args.n_blocks

    cfg = load_config()
    _maybe_apply_tuned_params(cfg)
    params = {kk: vv for kk, vv in cfg["baseline_xgb"].items()
              if kk not in {"early_stopping_rounds"}}

    X, y = build_dataset(cfg)
    blocks = make_blocks(len(X), k)
    logger.info("K=%d blocks: %s", k, blocks)

    records = []
    for b in range(1, k):
        te_start, te_end = blocks[b]
        X_te = X.iloc[te_start:te_end]
        y_te = y.iloc[te_start:te_end]

        # WINDOWED — only the immediately preceding block
        w_start, w_end = blocks[b - 1]
        X_tr = X.iloc[w_start:w_end]
        y_tr = y.iloc[w_start:w_end]
        logger.info("windowed b=%d: train blocks[%d] (n=%d), eval blocks[%d] (n=%d)",
                    b, b - 1, len(X_tr), b, len(X_te))
        r = fit_and_score(X_tr, y_tr, X_te, y_te, params)
        r.update({"strategy": "windowed", "eval_block": b})
        records.append(r)

        # CUMULATIVE — all blocks strictly before b
        X_tr = X.iloc[0:te_start]
        y_tr = y.iloc[0:te_start]
        logger.info("cumulative b=%d: train blocks[0..%d) (n=%d), eval blocks[%d] (n=%d)",
                    b, b, len(X_tr), b, len(X_te))
        r = fit_and_score(X_tr, y_tr, X_te, y_te, params)
        r.update({"strategy": "cumulative", "eval_block": b})
        records.append(r)

        # GLOBAL BASELINE — the shipped model (trained on all 1.18M rows)
        r = score_global_baseline(X_te, y_te)
        r.update({"strategy": "global_baseline", "eval_block": b})
        records.append(r)

    df = pd.DataFrame(records)[["eval_block", "strategy", "auc", "mcc_at_0_5",
                                 "mcc_at_optimal", "n_train", "n_eval"]]
    print(df.to_string(index=False))
    out_csv = REPO_ROOT / "models" / "windowed_metrics.csv"
    df.to_csv(out_csv, index=False)
    logger.info("saved %s", out_csv)

    # Chart: AUC by eval block, three strategies
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for strat, color, marker in [
        ("windowed", "#1f77b4", "o"),
        ("cumulative", "#2ca02c", "s"),
        ("global_baseline", "#d62728", "^"),
    ]:
        sub = df[df["strategy"] == strat].sort_values("eval_block")
        ax1.plot(sub["eval_block"], sub["auc"], "-", color=color, marker=marker,
                 label=strat, linewidth=2, markersize=8)
        ax2.plot(sub["eval_block"], sub["mcc_at_optimal"], "-", color=color, marker=marker,
                 label=strat, linewidth=2, markersize=8)

    ax1.axhline(0.5, color="gray", linestyle=":", alpha=0.6, label="random (AUC=0.5)")
    ax1.set_ylabel("AUC on eval block", fontsize=11)
    ax1.set_title(f"Windowed vs cumulative vs global-baseline (K={k} chronological blocks)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    ax2.set_ylabel("MCC at optimal threshold", fontsize=11)
    ax2.set_xlabel("Eval block index (later = further in time)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    plt.tight_layout()
    out_png = REPO_ROOT / "docs" / "img" / "windowed_experiment.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    logger.info("saved %s", out_png)


if __name__ == "__main__":
    main()
