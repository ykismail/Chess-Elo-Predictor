# Chess Elo Predictor — Multi-Phase Pipeline Makefile
# ====================================================
# Orchestrates data loading, validation, transformation, and feature engineering.

.PHONY: help setup pipeline phase1 validate-phase1 phase2 validate-phase2 phase3 validate-phase3 phase4 train phase5 predict clean test test-unit test-integration test-coverage test-report dashboard mlflow-report lint format isort lint-all

PYTHON ?= python
CONFIG ?= configs/config.toml

# ────────────────────────────────────────────────────────────────────────────
# Help target
# ────────────────────────────────────────────────────────────────────────────
help:
	@echo "Chess Elo Predictor - Multi-Phase ML Pipeline"
	@echo "================================================================================"
	@echo "Setup & Install:"
	@echo "  make setup           - Install Poetry dependencies"
	@echo "Pipeline Phases:"
	@echo "  make phase1          - Phase 1: Load & Parse Raw Data"
	@echo "  make validate-phase1 - Validate parsed_data_uci, parsed_data_pgn"
	@echo "  make phase2          - Phase 2: Transform & Merge"
	@echo "  make validate-phase2 - Validate kaggle_merged"
	@echo "  make phase3          - Phase 3: Engineer Features"
	@echo "  make validate-phase3 - Validate merged_games"
	@echo "  make train           - Phase 4: Train Models"
	@echo "  make predict         - Phase 5: Make Predictions"
	@echo "  make pipeline        - Run entire pipeline (Phases 1-3 with validations)"
	@echo "Testing & Reports:"
	@echo "  make test            - Run all tests with coverage"
	@echo "  make test-unit       - Unit tests only"
	@echo "  make test-integration - Integration tests only"
	@echo "  make test-coverage   - Detailed coverage report"
	@echo "  make test-report     - Generate HTML test reports"
	@echo "Deployment:"
	@echo "  make dashboard       - Generate static HTML dashboard for GitHub Pages"
	@echo "  make mlflow-report   - Generate static MLflow experiment report"
	@echo "Code Quality:"
	@echo "  make lint            - Run flake8"
	@echo "  make format          - Format with black"
	@echo "  make isort           - Sort imports with isort"
	@echo "  make lint-all        - Run all lint checks"
	@echo "Utilities:"
	@echo "  make clean           - Remove generated files"

# ────────────────────────────────────────────────────────────────────────────
# Setup
# ────────────────────────────────────────────────────────────────────────────
setup:
	@echo "[SETUP] Installing dependencies with Poetry..."
	python -m poetry install
	@echo "[OK] Setup complete."

# ────────────────────────────────────────────────────────────────────────────
# PHASE 1 & Validation
# ────────────────────────────────────────────────────────────────────────────
phase1:
	@echo "[PHASE 1] Load & Parse Raw Data"
	@echo "================================================================================"
	@echo "- Downloading raw data sources (Kaggle, GitHub, Lichess)"
	@echo "- Parsing PGN/UCI files into structured DataFrames"
	@echo "- Outputs: parsed_data_uci.csv, parsed_data_pgn.csv, eco_openings.csv"
	python -m poetry run python src/data/data_loading.py
	@echo "[OK] Phase 1 complete."

validate-phase1:
	@echo "[VALIDATE] Phase 1 Outputs"
	@echo "================================================================================"
	@echo "- Validates: parsed_data_uci.csv, parsed_data_pgn.csv"
	python -m poetry run python src/data/validate.py --phase sources
	@echo "[OK] Phase 1 validation complete."

# ────────────────────────────────────────────────────────────────────────────
# PHASE 2 & Validation
# ────────────────────────────────────────────────────────────────────────────
phase2: phase1
	@echo "[PHASE 2] Transform & Merge"
	@echo "================================================================================"
	@echo "- ECO opening matching (PGN to opening codes)"
	@echo "- Stockfish feature extraction (engine evaluations)"
	@echo "- Inner-join merge: PGN + UCI + Stockfish"
	@echo "- Output: kaggle_merged.csv"
	python -m poetry run python src/data/transform.py
	@echo "[OK] Phase 2 complete."

validate-phase2:
	@echo "[VALIDATE] Phase 2 Outputs"
	@echo "================================================================================"
	@echo "- Validates: kaggle_merged.csv"
	python -m poetry run python src/data/validate.py --phase merged
	@echo "[OK] Phase 2 validation complete."

# ────────────────────────────────────────────────────────────────────────────
# PHASE 3 & Validation
# ────────────────────────────────────────────────────────────────────────────
phase3: phase2
	@echo "[PHASE 3] Engineer Features"
	@echo "================================================================================"
	@echo "- Feature engineering: Elo buckets, acl_gap, winner targets"
	@echo "- Lichess integration: Harmonise & merge Lichess data (load_lichess)"
	@echo "- Final outputs: games.csv (~23k Kaggle), merged_games.csv (~43k combined)"
	python -m poetry run python src/data/build_features.py
	@echo "[OK] Phase 3 complete."

validate-phase3:
	@echo "[VALIDATE] Phase 3 Outputs"
	@echo "================================================================================"
	@echo "- Validates: merged_games.csv"
	python -m poetry run python src/data/validate.py --phase features
	@echo "[OK] Phase 3 validation complete."

# ────────────────────────────────────────────────────────────────────────────
# PHASE 4 (Train)
# ────────────────────────────────────────────────────────────────────────────
phase4: phase3
	@echo "[PHASE 4] Train Models"
	@echo "================================================================================"
	@echo "- Training multiple models (Dummy, Logistic Regression, Random Forest, XGBoost, MLP)"
	@echo "- Both Track A and Track B feature sets"
	@echo "- GridSearchCV hyperparameter tuning with stratified k-fold CV"
	@echo "- Outputs: model_*.pkl, classification reports, model_comparison.csv"
	python -m poetry run python src/models/train.py
	@echo "[OK] Phase 4 complete."

train: phase4

# ────────────────────────────────────────────────────────────────────────────
# PHASE 5 (Predict)
# ────────────────────────────────────────────────────────────────────────────
phase5: phase4
	@echo "[PHASE 5] Make Predictions"
	@echo "================================================================================"
	@echo "- Loading best trained model (random_forest_track_b by default)"
	@echo "- Predicting on held-out test set"
	@echo "- Outputs: predictions_*.csv, classification metrics"
	python -m poetry run python src/models/predict.py
	@echo "[OK] Phase 5 complete."

predict: phase5

# ────────────────────────────────────────────────────────────────────────────
# Full Pipeline (Phases 1-3 with validations interleaved)
# ────────────────────────────────────────────────────────────────────────────
pipeline: clean phase1 validate-phase1 phase2 validate-phase2 phase3 validate-phase3 phase4 phase5
	@echo "================================================================================"
	@echo "  DATA PIPELINE COMPLETE"
	@echo "================================================================================"
	@echo "  Outputs:"
	@echo "  - games.csv              (~23k Kaggle games with Stockfish)"
	@echo "  - merged_games.csv       (~43k combined games)"
	@echo "  Ready for modeling & analysis!"
	@echo "================================================================================"

# ────────────────────────────────────────────────────────────────────────────
# Code Quality Checks
# ────────────────────────────────────────────────────────────────────────────
test:
	@echo "Running all unit and integration tests with coverage..."
	python -m poetry run pytest tests/ --tb=short -v --maxfail=999 --cov=src --cov-report=html:reports/coverage/html --cov-report=xml:reports/coverage/coverage.xml --cov-report=term-missing --junitxml=reports/junit-results.xml || true
	@echo "✓ Tests complete. Coverage report: reports/coverage/html/index.html"

test-unit:
	@echo "Running unit tests only (excluding integration tests)..."
	python -m poetry run pytest tests/test_*.py --ignore=tests/test_integration_pipeline.py -v --tb=short || true

test-integration:
	@echo "Running integration tests only..."
	python -m poetry run pytest tests/test_integration_pipeline.py -v --tb=short -s || true

test-coverage:
	@echo "Running tests with detailed coverage report..."
	python -m poetry run pytest tests/ --cov=src --cov-report=html:reports/coverage/html --cov-report=term-missing -v || true

test-report:
	@echo "Generating test execution and coverage reports..."
	python -m poetry run python scripts/generate_test_report.py
	@echo "✓ Reports generated in reports/test-results/"

dashboard:
	@echo "Generating static HTML dashboard..."
	python -m poetry run python scripts/generate_static_dashboard.py
	@echo "✓ Dashboard generated in dist/"
	@echo "  Open in browser: dist/index.html"

mlflow-report:
	@echo "Generating static MLflow reports..."
	python -m poetry run python scripts/generate_mlflow_report.py
	@echo "✓ MLflow report generated in dist/mlflow/"

lint:
	@echo "Running flake8 linter..."
	-python -m poetry run flake8 src/ scripts/

format:
	@echo "Formatting code with black..."
	-python -m poetry run black src/ scripts/ stockfish_preprocessing/

isort:
	@echo "Sorting imports with isort..."
	-python -m poetry run isort src/ scripts/ stockfish_preprocessing/

lint-all: lint format isort
	@echo "Code quality checks complete."

# ────────────────────────────────────────────────────────────────────────────
# Cleanup (Cross-platform)
# ────────────────────────────────────────────────────────────────────────────
clean:
	@echo "Cleaning generated files..."
	python -c "import shutil, os; [shutil.rmtree(d) for d in __import__('pathlib').Path('.').rglob('__pycache__')]"
	python -c "from pathlib import Path; [f.unlink() for f in Path('.').rglob('*.pyc')]"
	python -c "from pathlib import Path; [f.unlink() for f in Path('.').rglob('.DS_Store')]"
	@echo "✓ Cleanup complete."

# ────────────────────────────────────────────────────────────────────────────
# Default
# ────────────────────────────────────────────────────────────────────────────
.DEFAULT_GOAL := help