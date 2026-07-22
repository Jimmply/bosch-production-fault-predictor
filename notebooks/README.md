# Notebooks

- `01_eda.ipynb` — chunked EDA of the 14 GB dataset. Missing-rate per feature, class balance verification, station-timing distributions.
- `02_survival_framing.ipynb` — validates that station timestamps encode transit times suitable for Cox regression.
- `03_baseline_vs_enriched.ipynb` — head-to-head: vanilla XGBoost vs XGBoost + station-attribution vs Cox survival model.
- `04_station_attribution.ipynb` — final SHAP heatmap by station group for the README hero image.
