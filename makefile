# Chess Elo Predictor — Multi-Phase Pipeline Makefile
# ====================================================
# Orchestrates data loading, validation, transformation, and feature engineering.

.PHONY: help setup pipeline phase1 phase2 phase3 phase4 validate clean test lint format isort

PYTHON ?= python
CONFIG ?= configs/config.toml

# ────────────────────────────────────────────────────────────────────────────
# Help target
# ────────────────────────────────────────────────────────────────────────────
help:
	@echo Chess Elo Predictor - Multi-Phase Pipeline
	@echo ================================================================================
	@echo Setup ^& Install:
	@echo   make setup           - Install Poetry dependencies
	@echo Pipeline Phases:
	@echo   make phase1          - Phase 1: Data Loading ^& Parsing
	@echo   make phase3          - Phase 3: Transformation ^& Merge
	@echo   make phase4          - Phase 4: Feature Engineering ^& Integration
	@echo   make pipeline        - Run entire pipeline with validations
	@echo Validation:
	@echo   make validate-phase1 - Validate Phase 1 outputs
	@echo   make validate-phase3 - Validate Phase 3 outputs
	@echo   make validate-phase4 - Validate Phase 4 outputs
	@echo   make validate        - Run all validations
	@echo Code Quality:
	@echo   make test            - Run pytest
	@echo   make lint            - Run flake8
	@echo   make format          - Format with black
	@echo   make isort           - Sort imports with isort
	@echo   make lint-all        - Run all lint checks
	@echo Utilities:
	@echo   make clean           - Remove generated files

# ────────────────────────────────────────────────────────────────────────────
# Setup
# ────────────────────────────────────────────────────────────────────────────
setup:
	@echo [SETUP] Installing dependencies with Poetry...
	python -m poetry install
	@echo [OK] Setup complete.

# ────────────────────────────────────────────────────────────────────────────
# PHASE 1: Data Loading & Parsing
# ────────────────────────────────────────────────────────────────────────────
phase1:
	@echo [PHASE 1] Data Loading ^& Parsing
	@echo ================================================================================
	@echo - Downloading raw data sources (Kaggle, GitHub, Lichess)
	@echo - Parsing PGN/UCI files into structured DataFrames
	@echo - Outputs: parsed_data_uci.csv, parsed_data_pgn.csv, eco_openings.csv
	python -m poetry run python src/data/data_loading.py
	@echo [OK] Phase 1 complete.

# ────────────────────────────────────────────────────────────────────────────
# PHASE 2: Data Validation (read-only)
# ────────────────────────────────────────────────────────────────────────────
validate-phase1:
	@echo [PHASE 2] Validation - Phase 1 Outputs
	@echo ================================================================================
	@echo - Validates: parsed_data_uci.csv, parsed_data_pgn.csv
	python -m poetry run python src/data/validate.py --phase sources
	@echo [OK] Phase 1 validation complete.

validate-phase3:
	@echo [PHASE 2] Validation - Phase 3 Outputs
	@echo ================================================================================
	@echo - Validates: kaggle_merged.csv
	python -m poetry run python src/data/validate.py --phase merged
	@echo [OK] Phase 3 validation complete.

validate-phase4:
	@echo [PHASE 2] Validation - Phase 4 Outputs
	@echo ================================================================================
	@echo - Validates: merged_games.csv
	python -m poetry run python src/data/validate.py --phase features
	@echo [OK] Phase 4 validation complete.

# ────────────────────────────────────────────────────────────────────────────
# PHASE 3: Transformation ^& Merge
# ────────────────────────────────────────────────────────────────────────────
phase3: phase1
	@echo [PHASE 3] Transformation ^& Merge
	@echo ================================================================================
	@echo - ECO opening matching (PGN to opening codes)
	@echo - Stockfish feature extraction (engine evaluations)
	@echo - Inner-join merge: PGN + UCI + Stockfish
	@echo - Output: kaggle_merged.csv
	python -m poetry run python src/data/transform.py
	@echo [OK] Phase 3 complete.

# ────────────────────────────────────────────────────────────────────────────
# PHASE 4: Feature Engineering ^& Integration
# ────────────────────────────────────────────────────────────────────────────
phase4: phase3
	@echo [PHASE 4] Feature Engineering ^& Integration
	@echo ================================================================================
	@echo - Feature engineering: Elo buckets, acl_gap, winner targets
	@echo - Lichess integration: Harmonise ^& merge Lichess data (load_lichess)
	@echo - Final outputs: games.csv (~23k Kaggle), merged_games.csv (~43k combined)
	python -m poetry run python src/data/build_features.py
	@echo [OK] Phase 4 complete.

# ────────────────────────────────────────────────────────────────────────────
# Full Pipeline (Phases 1-4 with validation after each phase)
# ────────────────────────────────────────────────────────────────────────────
pipeline: clean phase1 validate-phase1 phase3 validate-phase3 phase4 validate-phase4
	@echo ================================================================================
	@echo   ✓ PIPELINE COMPLETE
	@echo ================================================================================
	@echo   Outputs:
	@echo   - games.csv              (~23k Kaggle games with Stockfish)
	@echo   - merged_games.csv       (~43k combined games)
	@echo   Ready for modeling ^& analysis!
	@echo ================================================================================

# ────────────────────────────────────────────────────────────────────────────
# Data Validation (run all validations)
# ────────────────────────────────────────────────────────────────────────────
validate: validate-phase1 validate-phase3 validate-phase4
	@echo [OK] All validations complete.

# ────────────────────────────────────────────────────────────────────────────
# Code Quality Checks
# ────────────────────────────────────────────────────────────────────────────
test:
	@echo "Running tests with pytest..."
	-python -m poetry run pytest tests/

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
	@echo "✓ All code quality checks complete."

# ────────────────────────────────────────────────────────────────────────────
# Cleanup
# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────
# Cleanup (Windows-compatible)
# ────────────────────────────────────────────────────────────────────────────
clean:
	@echo "Cleaning generated files..."
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '.DS_Store' -delete
	@echo "✓ Cleanup complete."

# ────────────────────────────────────────────────────────────────────────────
# Default
# ────────────────────────────────────────────────────────────────────────────
.DEFAULT_GOAL := help
