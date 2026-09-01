"""Process-drift diagnostic — was the time-aware AUC collapse caused by drift?

Sorts all parts by transit_time_first, bins into N sequential time windows,
and reports per-window:
  - number of parts
  - positive (defect) rate
  - mean of the top-K SHAP-attributed station features

If any of those numbers move systematically across windows, the physical
process drifted over the training-set time span — which would explain why
a model trained on early parts cannot generalize to later ones.

Writes a 3-panel chart to docs/img/drift_analysis.png and prints a short
summary you can paste into the README.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data_loader import load_config, resolve_paths
from features import station_aggregates, transit_time_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

N_WINDOWS = 10
TOP_K_FEATURES = 5


def main() -> None:
    cfg = load_config()
    paths = resolve_paths(cfg)
    id_col = cfg["loading"]["id_col"]
    target_col = cfg["loading"]["target_col"]

    logger.info("loading train_numeric.parquet")
    numeric = pd.read_parquet(paths.parquet_dir / "train_numeric.parquet")
    logger.info("loading train_date.parquet")
    date_df = pd.read_parquet(paths.parquet_dir / "train_date.parquet")

    # Positional join — the two files come from the same source in the same order
    date_df = date_df.set_index(id_col).loc[numeric[id_col].values].reset_index()

    transit = transit_time_features(date_df.drop(columns=[id_col]))
    df = pd.DataFrame({
        "transit_time_first": transit["transit_time_first"].values,
        target_col: numeric[target_col].astype(int).values,
    })

    # Pick the top-K SHAP-attributed stations, look at their mean value per window
    attr = pd.read_parquet(REPO_ROOT / "models" / "station_attribution.parquet")
    top_stations = attr.head(TOP_K_FEATURES)["station"].tolist()
    logger.info("top-%d stations to track: %s", TOP_K_FEATURES, top_stations)

    features = numeric.drop(columns=[target_col, id_col])
    agg = station_aggregates(features)
    for st in top_stations:
        col = f"{st}__mean"
        if col in agg.columns:
            df[col] = agg[col].fillna(0.0).values

    # Bin by transit_time_first quantile — equal-sized time windows
    df = df.dropna(subset=["transit_time_first"]).sort_values("transit_time_first").reset_index(drop=True)
    df["window"] = pd.qcut(df["transit_time_first"], q=N_WINDOWS, labels=False, duplicates="drop")

    per_window = df.groupby("window").agg(
        n=("transit_time_first", "size"),
        positive_rate=(target_col, "mean"),
        t_first_min=("transit_time_first", "min"),
        t_first_max=("transit_time_first", "max"),
    )
    for st in top_stations:
        col = f"{st}__mean"
        if col in df.columns:
            per_window[col] = df.groupby("window")[col].mean()

    print(per_window.to_string())

    # Save numeric summary alongside model artifacts for reproducibility
    per_window.to_csv(REPO_ROOT / "models" / "drift_summary.csv")

    # 3-panel chart
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(per_window.index, per_window["positive_rate"] * 100, "o-", color="#d62728")
    axes[0].axhline(per_window["positive_rate"].mean() * 100, color="gray", linestyle="--", alpha=0.6,
                    label=f"overall mean = {per_window['positive_rate'].mean() * 100:.2f}%")
    axes[0].set_ylabel("defect rate (%)")
    axes[0].set_title("Positive-rate drift across sequential time windows")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(per_window.index, per_window["n"], color="#1f77b4", alpha=0.7)
    axes[1].set_ylabel("parts per window")
    axes[1].set_title("Window sizes (equal by count)")
    axes[1].grid(True, alpha=0.3)

    for st in top_stations:
        col = f"{st}__mean"
        if col in per_window.columns:
            v = per_window[col].values
            # z-score-normalize each series so different scales fit on one axis
            v_norm = (v - np.mean(v)) / (np.std(v) + 1e-9)
            axes[2].plot(per_window.index, v_norm, "o-", label=st)
    axes[2].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[2].set_ylabel("z-scored per-window mean")
    axes[2].set_xlabel("time window (sequential, equal-count)")
    axes[2].set_title(f"Top-{TOP_K_FEATURES} station-mean features across windows")
    axes[2].legend(loc="upper right", ncol=2)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out = REPO_ROOT / "docs" / "img" / "drift_analysis.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("saved %s", out)

    # Summary numbers to paste into the README
    pr_min, pr_max = per_window["positive_rate"].min(), per_window["positive_rate"].max()
    pr_ratio = pr_max / pr_min if pr_min > 0 else float("inf")
    print(f"\n== summary ==")
    print(f"positive-rate range across {N_WINDOWS} windows: "
          f"{pr_min * 100:.3f}%..{pr_max * 100:.3f}%  (ratio {pr_ratio:.2f}x)")
    # For the standardised feature means: report absolute range, and whether
    # the sign flips across windows. Relative % is meaningless when the
    # overall mean is near zero.
    for st in top_stations:
        col = f"{st}__mean"
        if col in per_window.columns:
            v = per_window[col].dropna()
            if len(v):
                sign_flips = int(((np.sign(v).diff().abs()) > 0).sum())
                print(f"{st}: range={v.min():+.4f}..{v.max():+.4f}, "
                      f"span={v.max() - v.min():.4f}, sign flips across windows={sign_flips}")


if __name__ == "__main__":
    main()
