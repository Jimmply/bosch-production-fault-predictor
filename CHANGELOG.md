# Changelog

Notable changes across iterations. Most-recent first.

## 2026-09-13 — drift-aware features (marginal, honestly)

- Wired `drift_features` into `train.py` behind a config flag. Added
  `src/drift_features.py` + tests for rolling z-scores that use only past
  values (`shift(1).rolling()`, no lookahead).
- Full-data time-aware retrain with drift features enabled. All 5 drift-z
  columns are picked up by XGBoost (gain 513–1301) but CV AUC only moved
  +0.003. Conclusion: the drift is in P(y|x), not P(x); rolling z-scores
  on x can't fix that. Two remaining branches noted in the roadmap.

## 2026-09-08 — windowed retraining doesn't help

- New `scripts/windowed_experiment.py`. K=5 chronological blocks. For
  each eval block: fit windowed (previous block only), cumulative (all
  earlier blocks), and evaluate the shipped global model.
- Neither windowed nor cumulative beats AUC ≈ 0.60 on truly held-out
  blocks. The global-baseline shows AUC ≈ 0.90 on the same blocks —
  quantifying ~0.30 AUC points of stratified-k-fold leakage.

## 2026-08-31 — process-drift diagnostic

- `scripts/drift_analysis.py`. Bins 1.18M parts by transit_time_first
  into K sequential windows and reports per-window defect rate + top-
  SHAP-station means.
- Finding: defect rate ranges 0.247%..0.982% across 10 windows (~4×
  swing). Top-station means flip sign 2–4× per feature. Drift is real.

## 2026-08-28 — time-aware CV, honest results

- Added `--split-strategy time` to `train.py`. Sorts by
  `transit_time_first` and uses `TimeSeriesSplit`.
- Full-data comparison: stratified vs time-aware. CV AUC 0.717 → 0.563,
  fold std grew 10×. Time-aware is now the shipped default. Stratified
  numbers were leaking future information backward through the shuffle.

## 2026-08-20 — Optuna tuning end-to-end

- Ran 15-trial TPE on 150k stratified sample via `scripts/tune_xgb.py`.
- Best MCC 0.188 on the tuning sample. Full-data retrain via the loader
  in `train.py` gave slight AUC gain and flat MCC. Deprioritized further
  tuning; model is not the bottleneck.

## 2026-07-28 — makefile + tooling

- `Makefile` with `setup / download / train / tune / cox / drift /
  windowed / dashboard / test / clean` targets. `train-time`,
  `train-strat`, `train-nodrift` shortcuts for A/B experiments.
  `results` target regenerates the analysis charts.

## 2026-07-22 — initial baseline

- XGBoost + engineered features on the full 1.18M-row Bosch training
  set. Station-attribution SHAP (Line 3 dominance, 11 stations for 70%
  of |SHAP|). Cox proportional-hazards on the top-30 SHAP-attributed
  stations (concordance 0.771). Streamlit dashboard, CI, tests, README.
