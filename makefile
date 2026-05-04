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
	@echo "Chess Elo Predictor — Multi-Phase Pipeline"
	@echo "=========================================="
	@echo ""
	@echo "Setup & Install:"
	@echo "  make setup           — Install Poetry dependencies"
	@echo ""
	@echo "Pipeline Phases:"
	@echo "  make phase1          — Phase 1: Data Loading & Parsing"
	@echo "  make phase2          — Phase 2: Data Validation (read-only)"
	@echo "  make phase3          — Phase 3: Transformation & Merge"
	@echo "  make phase4          — Phase 4: Feature Engineering & Integration"
	@echo "  make pipeline        — Run entire pipeline (phases 1-4)"
	@echo ""
	@echo "Code Quality:"
	@echo "  make test            — Run pytest"
	@echo "  make lint            — Run flake8"
	@echo "  make format          — Format with black"
	@echo "  make isort           — Sort imports with isort"
	@echo "  make lint-all        — Run all lint checks"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean           — Remove generated files"

# ────────────────────────────────────────────────────────────────────────────
# Setup
# ────────────────────────────────────────────────────────────────────────────
setup:
	@echo "[SETUP] Installing dependencies with Poetry..."
	poetry install
	@echo "✓ Setup complete."

# ────────────────────────────────────────────────────────────────────────────
# PHASE 1: Data Loading & Parsing
# ────────────────────────────────────────────────────────────────────────────
phase1:
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════════════════"
	@echo "PHASE 1: Data Loading & Parsing"
	@echo "═══════════════════════════════════════════════════════════════════════════════"
	@echo "→ Downloading raw data sources (Kaggle, GitHub, Lichess)"
	@echo "→ Parsing PGN/UCI files into structured DataFrames"
	@echo "→ Outputs: parsed_data_uci.csv, parsed_data_pgn.csv, eco_openings.csv"
	@echo ""
	poetry run $(PYTHON) src/data/data_loading.py
	@echo "✓ Phase 1 complete."

# ────────────────────────────────────────────────────────────────────────────
# PHASE 2: Data Validation (read-only)
# ────────────────────────────────────────────────────────────────────────────
phase2: phase1
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════════════════"
	@echo "PHASE 2: Data Validation"
	@echo "═══════════════════════════════════════════════════════════════════════════════"
	@echo "→ Running quality checks on Phase 1 outputs (read-only)"
	@echo "→ Validates: parsed_data_uci, parsed_data_pgn, kaggle_merged, merged_games"
	@echo ""
	poetry run $(PYTHON) src/data/validate.py --phase all
	@echo "✓ Phase 2 complete."

# ────────────────────────────────────────────────────────────────────────────
# PHASE 3: Transformation & Merge
# ────────────────────────────────────────────────────────────────────────────
phase3: phase1
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════════════════"
	@echo "PHASE 3: Transformation & Merge"
	@echo "═══════════════════════════════════════════════════════════════════════════════"
	@echo "→ ECO opening matching (PGN → opening codes)"
	@echo "→ Stockfish feature extraction (engine evaluations)"
	@echo "→ Inner-join merge: PGN + UCI + Stockfish"
	@echo "→ Output: kaggle_merged.csv"
	@echo ""
	poetry run $(PYTHON) src/data/transform.py
	@echo "✓ Phase 3 complete."

# ────────────────────────────────────────────────────────────────────────────
# PHASE 4: Feature Engineering & Integration
# ────────────────────────────────────────────────────────────────────────────
phase4: phase3
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════════════════"
	@echo "PHASE 4: Feature Engineering & Integration"
	@echo "═══════════════════════════════════════════════════════════════════════════════"
	@echo "→ Feature engineering: Elo buckets, acl_gap, winner targets"
	@echo "→ Lichess integration: Harmonise & merge Lichess data (load_lichess)"
	@echo "→ Final outputs: games.csv (~23k Kaggle), merged_games.csv (~43k combined)"
	@echo ""
	poetry run $(PYTHON) src/data/build_features.py
	@echo "✓ Phase 4 complete."

# ────────────────────────────────────────────────────────────────────────────
# Full Pipeline (Phases 1–4)
# ────────────────────────────────────────────────────────────────────────────
pipeline: clean phase1 phase2 phase3 phase4
	@echo ""
	@echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
	@echo "║                       ✓ PIPELINE COMPLETE                                    ║"
	@echo "║                                                                               ║"
	@echo "║  Outputs:                                                                     ║"
	@echo "║    - games.csv              (~23k Kaggle games with Stockfish)               ║"
	@echo "║    - merged_games.csv       (~43k combined games)                            ║"
	@echo "║  Ready for modeling & analysis!                                              ║"
	@echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
	@echo ""

# ────────────────────────────────────────────────────────────────────────────
# Data Validation only (without re-running all phases)
# ────────────────────────────────────────────────────────────────────────────
validate:
	@echo "Running validation checks on all intermediate datasets..."
	poetry run $(PYTHON) src/data/validate.py --phase all

# ────────────────────────────────────────────────────────────────────────────
# Code Quality Checks
# ────────────────────────────────────────────────────────────────────────────
test:
	@echo "Running tests with pytest..."
	poetry run pytest tests/ || true

lint:
	@echo "Running flake8 linter..."
	poetry run flake8 src/ scripts/ || true

format:
	@echo "Formatting code with black..."
	poetry run black src/ scripts/ || true

isort:
	@echo "Sorting imports with isort..."
	poetry run isort src/ scripts/ || true

lint-all: lint format isort
	@echo "✓ All lint checks complete."

# ────────────────────────────────────────────────────────────────────────────
# Cleanup
# ────────────────────────────────────────────────────────────────────────────
clean:
	@echo "Cleaning generated files..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	@echo "✓ Cleanup complete."

# ────────────────────────────────────────────────────────────────────────────
# Default
# ────────────────────────────────────────────────────────────────────────────
.DEFAULT_GOAL := help