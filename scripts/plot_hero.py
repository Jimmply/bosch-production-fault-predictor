"""Regenerate the top-20-stations hero chart from the shipped model artifacts.

Reads:
  - models/metrics.yaml
  - models/station_attribution.parquet
  - models/cox_concordance.txt   (optional; skipped if absent)

Writes docs/img/station_attribution.png.

Split out of train.py so `make results` can regen the hero without a full
training run.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from plot_utils import LINE_PALETTE  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main(top_n: int = 20) -> None:
    attr = pd.read_parquet(REPO_ROOT / "models" / "station_attribution.parquet").head(top_n)
    with open(REPO_ROOT / "models" / "metrics.yaml") as f:
        metrics = yaml.safe_load(f)

    concordance_path = REPO_ROOT / "models" / "cox_concordance.txt"
    concordance = float(concordance_path.read_text().strip()) if concordance_path.exists() else None

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [LINE_PALETTE[l] for l in attr["line"]]
    ax.barh(range(len(attr)), attr["share_of_total"] * 100, color=colors)
    ax.set_yticks(range(len(attr)))
    ax.set_yticklabels(attr["station"])
    ax.invert_yaxis()
    ax.set_xlabel("Share of total |SHAP| (%)", fontsize=11)

    title = (
        f"Top {top_n} stations — tuned XGBoost + drift features + time-aware CV\n"
        f"CV MCC={metrics['mcc_mean']:.3f}±{metrics['mcc_std']:.3f}  "
        f"AUC={metrics['auc_mean']:.3f}"
    )
    if concordance is not None:
        title += f"  Cox concordance={concordance:.3f}"
    ax.set_title(title, fontsize=11)

    ax.legend(
        handles=[Patch(facecolor=LINE_PALETTE[i], label=f"Line {i}") for i in sorted(LINE_PALETTE)],
        loc="lower right",
    )
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    out = REPO_ROOT / "docs" / "img" / "station_attribution.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("saved %s", out)


if __name__ == "__main__":
    main()
