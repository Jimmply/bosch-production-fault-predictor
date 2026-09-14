.PHONY: setup download train train-sample train-time train-stratified train-nodrift tune cox drift windowed results dashboard test clean lint help

PYTHON := .venv/bin/python
STREAMLIT := .venv/bin/streamlit
PYTEST := .venv/bin/pytest

help:
	@echo "targets:"
	@echo "  make setup       create venv and install deps"
	@echo "  make download    fetch and reencode Bosch data (needs Kaggle token)"
	@echo "  make train       train baseline XGBoost + SHAP attribution (uses split.strategy from config)"
	@echo "  make train-time  train with time-aware TimeSeriesSplit (walk-forward)"
	@echo "  make train-strat train with stratified k-fold (leaks future info — for comparison only)"
	@echo "  make train-nodrift  train without drift features (for the drift a/b comparison)"
	@echo "  make tune        run optuna hyperparameter search -> config/tuned_params.yaml"
	@echo "  make cox         fit Cox model on top-30 SHAP stations"
	@echo "  make drift       per-window drift diagnostic -> docs/img/drift_analysis.png"
	@echo "  make windowed    windowed vs cumulative vs global-baseline experiment"
	@echo "  make results     regenerate every analysis chart in docs/img/ (~10 min)"
	@echo "  make dashboard   launch Streamlit dashboard"
	@echo "  make test        pytest"
	@echo "  make clean       remove venv, caches, models"

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install pytest pytest-cov

download:
	$(PYTHON) scripts/download_data.py

train:
	$(PYTHON) scripts/train.py

train-sample:
	$(PYTHON) scripts/train.py --sample-n 300000

train-time:
	$(PYTHON) scripts/train.py --split-strategy time

train-strat:
	$(PYTHON) scripts/train.py --split-strategy stratified

train-nodrift:
	$(PYTHON) scripts/train.py --no-drift-features

tune:
	$(PYTHON) scripts/tune_xgb.py --n-trials 15 --sample-n 150000

cox:
	$(PYTHON) scripts/fit_cox.py

drift:
	$(PYTHON) scripts/drift_analysis.py

windowed:
	$(PYTHON) scripts/windowed_experiment.py --n-blocks 5

results: drift windowed
	$(PYTHON) scripts/plot_attribution_by_line.py
	@echo "all analysis charts regenerated -> docs/img/"

dashboard:
	$(STREAMLIT) run src/app.py

test:
	$(PYTEST) tests/ -v

lint:
	$(PYTHON) -m ruff check src tests scripts

clean:
	rm -rf .venv .pytest_cache .ruff_cache __pycache__ src/__pycache__ tests/__pycache__ scripts/__pycache__
	find . -name "*.pyc" -delete
