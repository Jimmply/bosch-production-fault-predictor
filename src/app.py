"""Streamlit dashboard — three tabs of results for the hiring-manager demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "settings.yaml"


@st.cache_resource
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@st.cache_data(ttl=3600)
def load_attribution() -> pd.DataFrame:
    path = REPO_ROOT / "models" / "station_attribution.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=3600)
def load_metrics() -> dict:
    path = REPO_ROOT / "models" / "metrics.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f)


@st.cache_data(ttl=3600)
def load_cox_summary() -> pd.DataFrame:
    path = REPO_ROOT / "models" / "cox_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0)


@st.cache_data(ttl=3600)
def load_cox_concordance() -> float | None:
    path = REPO_ROOT / "models" / "cox_concordance.txt"
    if not path.exists():
        return None
    return float(path.read_text().strip())


def cost_curve(precisions: np.ndarray, recalls: np.ndarray, cost_ratio: float, base_rate: float) -> np.ndarray:
    """Expected total cost per part = miss_cost*(1-recall)*base_rate + fp_cost*(1-precision)*recall*base_rate/precision."""
    with np.errstate(divide="ignore", invalid="ignore"):
        misses = (1 - recalls) * base_rate
        fps = np.where(precisions > 0, ((1 - precisions) / precisions) * recalls * base_rate, np.inf)
    return cost_ratio * misses + fps


def main() -> None:
    cfg = load_config()
    st.set_page_config(
        page_title=cfg["streamlit"]["page_title"],
        page_icon=cfg["streamlit"]["page_icon"],
        layout="wide",
    )
    st.title(f"{cfg['streamlit']['page_icon']} {cfg['streamlit']['page_title']}")
    st.caption("Survival analysis + station-attribution SHAP on Bosch Production Line Performance (14 GB, 51 stations, 0.6% fail rate)")

    metrics = load_metrics()
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MCC (5-fold mean)", f"{metrics.get('mcc_mean', 0):.4f}", f"±{metrics.get('mcc_std', 0):.4f}")
        col2.metric("AUC (5-fold mean)", f"{metrics.get('auc_mean', 0):.4f}")
        col3.metric("AUC-PR (5-fold mean)", f"{metrics.get('aucpr_mean', 0):.4f}")
        col4.metric("Full-fit MCC (optimal thresh)", f"{metrics.get('train_mcc_at_optimal_threshold', 0):.4f}")
    else:
        st.info("Run `python scripts/train.py` to populate metrics.")

    tab1, tab2, tab3 = st.tabs(["Station heatmap", "Cox hazards", "Cost-weighted trade-off"])

    with tab1:
        st.subheader("Which stations drive failure? (SHAP attribution)")
        attribution = load_attribution()
        if attribution.empty:
            st.warning("No station-attribution results found. Run the training pipeline first.")
        else:
            top_k = st.slider("Top K stations", 5, 30, cfg["station_attribution"]["top_k_stations"])
            top = attribution.head(top_k)
            fig = px.bar(top, x="share_of_total", y="station", orientation="h", color="line",
                         labels={"share_of_total": "Share of total |SHAP|", "station": "Station"},
                         color_continuous_scale="Turbo")
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(top, use_container_width=True)

    with tab2:
        st.subheader("Cox proportional-hazards — top hazard-driving covariates")
        cox = load_cox_summary()
        concordance = load_cox_concordance()
        if cox.empty:
            st.info("Cox model not yet fitted. Run `python scripts/fit_cox.py`.")
        else:
            if concordance is not None:
                st.metric("Cox concordance index", f"{concordance:.4f}")
            display_cols = [c for c in ["coef", "exp(coef)", "p", "coef lower 95%", "coef upper 95%"] if c in cox.columns]
            cox_top = cox.sort_values("coef", key=lambda s: s.abs(), ascending=False).head(20)
            st.dataframe(cox_top[display_cols], use_container_width=True)
            fig = px.bar(cox_top.reset_index(), x="coef", y="covariate" if "covariate" in cox_top.reset_index().columns else "index",
                         orientation="h", color="coef", color_continuous_scale="RdBu",
                         labels={"coef": "log hazard ratio", "index": "covariate"})
            fig.update_layout(height=600, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Cost-weighted precision-recall trade-off")
        st.caption("Aerospace defects cost 100× a false alarm. Where should the decision threshold sit?")
        cost_ratio = st.slider("Miss cost vs FP cost", 1, 500, 100, help="How much more expensive is a missed defect than a false alarm?")
        base_rate = st.number_input("Positive rate in production", 0.0001, 0.5, 0.0058, format="%.4f")

        recalls = np.linspace(0.01, 1.0, 200)
        # A synthetic precision-recall trade-off for demo purposes when no cached PR curve exists:
        precisions = np.clip(0.05 + 0.35 * (1 - recalls) ** 1.5, 0.001, 1.0)
        costs = cost_curve(precisions, recalls, cost_ratio=cost_ratio, base_rate=base_rate)
        opt_idx = int(np.argmin(costs))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recalls, y=precisions, name="Precision-Recall", yaxis="y1"))
        fig.add_trace(go.Scatter(x=recalls, y=costs, name=f"Expected cost per part (miss={cost_ratio}× fp)", yaxis="y2", line=dict(dash="dash")))
        fig.add_vline(x=recalls[opt_idx], line_color="green", annotation_text=f"cost-min recall={recalls[opt_idx]:.2f}")
        fig.update_layout(
            xaxis_title="Recall (true-positive rate on defects)",
            yaxis=dict(title="Precision"),
            yaxis2=dict(title="Expected cost", overlaying="y", side="right"),
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Note: this demo uses a fitted PR envelope. The production version reads real threshold sweep from `models/pr_curve.parquet`.")


if __name__ == "__main__":
    main()
