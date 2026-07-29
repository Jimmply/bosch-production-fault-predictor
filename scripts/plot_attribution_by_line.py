"""Aggregate SHAP attribution up from station-level to production-line-level.

Since Line 3 dominates the top-10 stations, it's worth seeing the full picture
across all four lines. Saves a compact bar chart for the README.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    attr = pd.read_parquet(REPO_ROOT / "models" / "station_attribution.parquet")
    per_line = attr.groupby("line", as_index=False).agg(
        share_of_total=("share_of_total", "sum"),
        n_stations=("station", "count"),
    ).sort_values("share_of_total", ascending=False)
    per_line["share_pct"] = per_line["share_of_total"] * 100
    per_line["label"] = per_line.apply(lambda r: f"Line {int(r['line'])}\n({int(r['n_stations'])} stations)", axis=1)

    print(per_line.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    palette = {0: "#1f77b4", 1: "#2ca02c", 2: "#ff7f0e", 3: "#d62728"}
    colors = [palette[l] for l in per_line["line"]]
    bars = ax.bar(per_line["label"], per_line["share_pct"], color=colors)
    for bar, pct in zip(bars, per_line["share_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Share of total |SHAP| (%)", fontsize=11)
    ax.set_title("Failure attribution rolled up to production line", fontsize=12)
    ax.set_ylim(0, max(per_line["share_pct"]) * 1.15)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = REPO_ROOT / "docs" / "img" / "attribution_by_line.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
