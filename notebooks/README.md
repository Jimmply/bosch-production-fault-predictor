# Notebooks

The end-to-end analysis currently lives in `scripts/` — the codebase was
written as importable modules first, with notebooks planned as a follow-up for
narrative/reviewer-facing walkthroughs.

Planned:
- `01_eda.ipynb` — chunked EDA with missing-rate heatmap and station-timing distributions.
- `02_survival_framing.ipynb` — walk through transit-time derivation + Cox concordance interpretation.
- `03_tuning_report.ipynb` — Optuna results once `scripts/tune_xgb.py` has been run.

For now, the reproducible entry points are:

```bash
make train        # baseline + SHAP attribution
make cox          # Cox proportional-hazards fit
make dashboard    # Streamlit
```
