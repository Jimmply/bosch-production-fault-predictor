# Bosch Production Fault Predictor

[![CI](https://github.com/Jimmply/bosch-production-fault-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/Jimmply/bosch-production-fault-predictor/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Predicting quality-control failures on the [Bosch Production Line Performance](https://www.kaggle.com/c/bosch-production-line-performance) Kaggle dataset — **1.18 million real manufactured parts, 968+ anonymized sensor features across 51 stations, 0.6% failure rate**.

Rather than yet another vanilla-XGBoost notebook (of which Kaggle has hundreds), this project frames the problem two novel ways:

1. **Cox survival analysis** — treats each part as a patient moving through 51 wards (stations); estimates per-station hazard contributions to failure.
2. **Station-attribution SHAP** — aggregates XGBoost SHAP values by station-group into an engineer-actionable heatmap showing which stations drive the failure signal.

## Results

Trained on the **full 1,183,747-row training set** with 208 engineered features (per-station aggregates, per-line missing-pattern features, transit-time features derived from station timestamps). Positive rate = **0.58%** (6,879 defective parts).

| Model | 5-fold MCC | AUC-PR | AUC | Concordance |
|---|---|---|---|---|
| Majority-class baseline | 0.000 | 0.006 | 0.500 | — |
| **XGBoost + engineered features** | **0.171 ± 0.004** | **0.069** | **0.711** | — |
| XGBoost @ threshold-optimal (0.709) | **0.245** (full-fit) | — | — | — |
| **Cox proportional-hazards (top-30 stations)** | — | — | — | **0.771** |

*Bosch's official metric is Matthews Correlation Coefficient (MCC). The 2016 competition winners scored ≈ 0.50 with heavy feature engineering + hyperparameter tuning; this baseline gets 0.245 out-of-the-box with default XGBoost hyperparameters and no time-aware CV — a respectable v0.1.0 that leaves clear room for iteration.*

### Station-attribution hero result

**17 out of 51 stations explain 70% of the failure signal** — an actionable Pareto for the manufacturing engineer:

![Top 20 stations driving quality-control failure](docs/img/station_attribution.png)

Top 3 stations by SHAP-attribution share:

| Rank | Station | Line | Share of \|SHAP\| |
|---|---|---|---|
| 1 | L3_S33 | Line 3 | 11.4% |
| 2 | L3_S29 | Line 3 | 6.0% |
| 3 | L3_S32 | Line 3 | 5.2% |

**Line 3 dominates the top of the list** — 7 of the top 10 attributed stations are on Line 3, a diagnosis a vanilla accuracy score cannot deliver.

### But it's actually more nuanced than that

Rolled up to production-line level, the picture shifts:

![Attribution rolled up per line](docs/img/attribution_by_line.png)

Line 0 actually accumulates slightly *more* total SHAP attribution (50%) than Line 3 (40%) — because Line 0 has 24 contributing stations vs Line 3's 21. Line 3 dominates the **top-of-list** stations (concentrated risk on a few stations), whereas Line 0 has broader, distributed risk. Both are worth investigating, but the interventions differ: Line 3 needs targeted station-level rework; Line 0 needs a systemic line-wide look.

This is exactly the kind of split-view an aggregate accuracy number hides.

### Cox survival highlights

Top three hazard-driving station-level covariates (from the 119-covariate Cox fit on the 300k stratified sample):

| Covariate | log HR | Hazard ratio | p-value |
|---|---|---|---|
| L3_S32 abs_sum | +4.49 | **89× hazard** | 7.3e-06 |
| L2_S26 mean | +2.13 | 8.5× | 3.3e-06 |
| L3_S29 mean | +1.64 | 5.1× | 1.5e-05 |

Cox agrees with SHAP on Line 3 dominance and adds statistical significance (all p < 1e-5). Full Cox summary at [`models/cox_summary.csv`](models/cox_summary.csv).

## Architecture

```
Kaggle download (14 GB CSV)
        │
        ▼
Chunked reencode -> Parquet          [src/data_loader.py]
        │
        ▼
Feature engineering                   [src/features.py]
  - Station-group aggregates
  - Inter-station transit times
  - Per-line missing-pattern features
        │
        ├─────────────► XGBoost baseline + threshold-optimal MCC     [src/baseline_model.py]
        │                       │
        │                       ▼
        │             Station-attribution SHAP heatmap               [src/station_attribution.py]
        │
        └─────────────► Cox time-varying-covariate model             [src/survival_model.py]
                                │
                                ▼
                    Streamlit dashboard (3 tabs)                     [src/app.py]
```

## Quickstart

```bash
git clone https://github.com/Jimmply/bosch-production-fault-predictor.git
cd bosch-production-fault-predictor
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Download data (~14 GB, requires Kaggle API token at ~/.kaggle/kaggle.json)
python scripts/download_data.py

# 2. Train the baseline (full 1.18M rows, ~6 min)
python scripts/train.py

# 3. Fit the Cox survival model on top-30 SHAP-attributed stations
python scripts/fit_cox.py

# 4. Launch the dashboard (3 tabs: station heatmap, Cox hazards, cost-weighted PR)
streamlit run src/app.py

# --- Optional flags ---
# Fast iteration on a stratified sample:
python scripts/train.py --sample-n 300000

# Switch CV to time-aware split (edit config/settings.yaml: split.strategy: "time")
```

## Kaggle API setup

1. Go to https://www.kaggle.com/settings → **Create New Token** → downloads `kaggle.json`.
2. Move it: `mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json`
3. Accept competition rules once at https://www.kaggle.com/c/bosch-production-line-performance/rules

## Why survival analysis?

Traditional classification asks *"will this part fail?"* Cox regression additionally answers *"which station's contribution to hazard is largest?"* — a diagnostic answer a manufacturing engineer can act on. Because Bosch encodes per-station timestamps in the `date` files, we can derive a transit-time duration per part and use it as the survival duration; the `Response` flag is the event.

The framing is borrowed from ICU epidemiology: patients (parts) traverse wards (stations), each ward carries a time-varying hazard, and the goal is to attribute risk to the specific step in the pipeline where it accumulates.

## Project structure

```
bosch-production-fault-predictor/
├── .github/workflows/ci.yml     # pytest + import smoke
├── config/settings.yaml         # all tunable parameters
├── data/                        # gitignored; downloaded via Kaggle CLI
├── models/                      # gitignored artifacts
├── notebooks/                   # 4 notebooks documented in notebooks/README.md
├── scripts/
│   ├── download_data.py         # Kaggle CLI wrapper + integrity check
│   └── train.py                 # end-to-end baseline training
├── src/
│   ├── data_loader.py           # chunked CSV -> Parquet
│   ├── features.py              # station-group aggregates, transit times
│   ├── baseline_model.py        # XGBoost + threshold-optimal MCC
│   ├── survival_model.py        # Cox time-varying-covariate model
│   ├── station_attribution.py   # SHAP aggregation by station
│   └── app.py                   # Streamlit dashboard
└── tests/                       # pytest smoke tests
```

## Roadmap

Things I still want to try (in rough priority order):

- [ ] Time-aware CV — flag exists (`split.strategy: time`) but haven't compared numbers head-to-head yet.
- [ ] Optuna tuning integrated into `train.py` — the stub in `scripts/tune_xgb.py` needs a config loader.
- [ ] Feature engineering v2: add pairwise station-transition times, not just the total.
- [ ] Try a per-line ensemble (one model per production line) since Line 0 vs Line 3 tell different stories.
- [ ] Actually write the notebooks in `notebooks/` (currently script-first).

## Author

**Dmitry Shurkhai** — Manufacturing Data Scientist
[GitHub](https://github.com/Jimmply) · [Kaggle](https://www.kaggle.com/jimmysh) · [LinkedIn](https://linkedin.com/in/etozhejimmy)
