# Chess Elo Predictor — Data Pipeline Documentation

## Overview

This document describes the **4-phase data pipeline** orchestrated by the `makefile`. The pipeline automates data loading, validation, transformation, and feature engineering for the Chess Elo Predictor project.

---

## Quick Start

### Prerequisites
- Python 3.14.4
- Poetry (for dependency management)
- Kaggle API credentials (for data download)

### Install Dependencies
```powershell
make setup
```

### Run Full Pipeline
```powershell
make pipeline
```

---

## Pipeline Architecture

The pipeline consists of **4 sequential phases**, each with specific responsibilities:

```
Phase 1 (Data Loading)
    ↓
Phase 2 (Validation) — read-only checks
    ↓
Phase 3 (Transformation) — ECO matching, Stockfish extraction, merging
    ↓
Phase 4 (Feature Engineering) — Elo buckets, target variables, Lichess integration
    ↓
Final Outputs: games.csv (~23k), merged_games.csv (~43k)
```

---

## Phase Details

### **Phase 1: Data Loading & Parsing**

**Command:** `make phase1`

**Responsibilities:**
- Download raw data from Kaggle (competition and datasets)
- Download ECO opening database from GitHub (Lichess)
- Parse PGN files into structured DataFrames
- Parse UCI notation files

**Entry Point:** `src/data/data_loading.py`

**Outputs:**
- `data/intermediate/parsed_data_uci.csv` — UCI move sequences per game
- `data/intermediate/parsed_data_pgn.csv` — Game metadata + SAN moves
- `data/external/eco_openings.csv` — ECO opening database

**Key Functions:**
- `acquire_data()` — Downloads all raw sources
- `parse_raw_files()` — Parses PGN/UCI into structured format

---

### **Phase 2: Data Validation (Read-Only)**

**Command:** `make phase2`

**Responsibilities:**
- Quality checks on Phase 1 outputs
- Validate data integrity, missing values, duplicates
- Generate validation reports

**Entry Point:** `src/data/validate.py`

**Outputs:**
- Console reports with data quality metrics
- No CSV files written (read-only phase)

**Key Functions:**
- `validate_sources()` — Validates Phase 1 outputs
- `validate_merged()` — Validates merged Kaggle data
- `validate_features()` — Validates final feature set

---

### **Phase 3: Transformation & Merge**

**Command:** `make phase3`

**Responsibilities:**
- ECO opening matching (longest-prefix matching on move sequences)
- Extract Stockfish evaluation features (ACL, blunders, mistakes, etc.)
- Inner-join merge: PGN + UCI + Stockfish

**Entry Point:** `src/data/transform.py`

**Outputs:**
- `data/intermediate/kaggle_merged.csv` — Fully merged Kaggle dataset

**Key Steps:**
1. Load Phase 1 outputs (parsed_data_uci, parsed_data_pgn)
2. Apply ECO matching to PGN games
3. Extract Stockfish features from raw stockfish.csv
4. Three-way inner-join merge

**Key Functions:**
- `apply_eco_matching()` — Match games to ECO codes
- `load_stockfish_features()` — Extract engine evaluation metrics
- `merge_kaggle_data()` — Merge three sources on event_id

---

### **Phase 4: Feature Engineering & Integration**

**Command:** `make phase4`

**Responsibilities:**
- Engineer derived features: Elo buckets, acl_gap, target variables
- Harmonise and integrate Lichess CSV dataset
- Create final output datasets

**Entry Point:** `src/data/build_features.py`

**Outputs:**
- `data/intermediate/games.csv` — Kaggle-only (~23k rows, with Stockfish)
- `data/intermediate/merged_games.csv` — Full combined dataset (~43k rows)

**Key Features Engineered:**
- `elo_bucket_white/black` — Categorical skill tier (Beginner–Master)
- `elo_bucket_white/black_categorical` — Ordinal encoding (0–4)
- `acl_gap` — Accuracy gap between players
- `winner_binary` — 1=White, 0=Black (excludes draws)
- `winner_multiclass` — 0=Black, 1=Draw, 2=White (target variable)

**Key Functions:**
- `engineer_features()` — Derive Elo buckets and target variables
- `integrate_datasets()` — Merge Lichess data with Kaggle pipeline
- `load_lichess()` — Harmonise Lichess CSV with pipeline schema

---

## Running Individual Phases

You can run individual phases without running the full pipeline:

```powershell
# Run only Phase 1
make phase1

# Run only Phase 2 (depends on Phase 1)
make phase2

# Run only Phase 3 (depends on Phase 1 outputs)
make phase3

# Run only Phase 4 (depends on Phase 3)
make phase4
```

---

## Code Quality Checks

The pipeline includes integrated code quality checks:

### Run All Checks
```powershell
make lint-all
```

### Individual Checks
```powershell
make lint       # Flake8 linter
make format     # Black formatter
make isort      # Import sorting
make test       # Pytest
```

---

## Validation & Cleanup

### Validate Data (Without Re-running Pipeline)
```powershell
make validate
```

### Clean Generated Files
```powershell
make clean      # Remove __pycache__, *.pyc, .DS_Store
```

---

## Configuration

Pipeline configuration is defined in `configs/config.toml`:

### Phase-Specific Configuration

**Phase 1 (data_loading):**
```toml
[phase1]
input_pgn_file = "data/raw/data.pgn"
input_uci_file = "data/raw/data_uci.pgn"
input_stockfish_file = "data/raw/stockfish.csv"
output_eco_csv = "data/external/eco_openings.csv"
```

**Phase 3 (transform):**
```toml
[phase3]
input_eco_csv = "data/external/eco_openings.csv"
output_kaggle_merged = "data/intermediate/kaggle_merged.csv"
```

**Phase 4 (build_features):**
```toml
[phase4]
input_lichess_file = "data/raw/chess_games.csv"
output_games_csv = "data/intermediate/games.csv"
output_merged_games = "data/intermediate/merged_games.csv"
```

**Target Variables:**
```toml
[target]
column = "elo_bucket_white_categorical"
classes = ["Beginner", "Intermediate", "Advanced", "Expert", "Master"]
elo_bins = [0, 1000, 1500, 2000, 2500, 9999]
```

---

## CI/CD Pipeline Integration

The pipeline is integrated into GitHub Actions (`.github/workflows/main.yml`):

1. **CI Job** — Triggered on push/PR:
   - `make setup` — Install dependencies
   - `make lint-all` — Run code quality checks
   - `make test` — Run pytest
   - `make pipeline` — Run all 4 phases

2. **CD Job** — Triggered on successful CI:
   - Builds Streamlit app with Stlite
   - Deploys to GitHub Pages

---

## Data Flow Diagram

```
Raw Data Sources (Kaggle, GitHub, Lichess)
    ↓
Phase 1: Parse & Download
    ↓ Outputs: parsed_data_*.csv, eco_openings.csv
    ↓
Phase 2: Validation (read-only)
    ↓ Reports console metrics (no new files)
    ↓
Phase 3: Merge & Transform
    ↓ Outputs: kaggle_merged.csv
    ↓
Phase 4: Feature Engineering
    ↓ Outputs: games.csv, merged_games.csv
    ↓
Ready for Modeling & Analysis
```

---

## File Structure

```
Chess-Elo-Predictor/
├── configs/
│   └── config.toml                    # Pipeline configuration
├── data/
│   ├── raw/                           # Downloaded raw files (not committed)
│   ├── intermediate/                  # Phase outputs (not committed)
│   └── external/                      # ECO database (not committed)
├── src/
│   ├── data/
│   │   ├── data_loading.py            # Phase 1 entry point
│   │   ├── validate.py                # Phase 2 entry point
│   │   ├── transform.py               # Phase 3 entry point
│   │   └── build_features.py          # Phase 4 entry point
│   │   └── load_data.py               # Utility functions
│   └── features/
│       └── build_features.py          # Feature engineering functions
├── .github/
│   └── workflows/
│       └── main.yml                   # CI/CD pipeline
├── makefile                           # Pipeline orchestration
├── pyproject.toml                     # Poetry dependencies
└── PIPELINE.md                        # This file
```

---

## Troubleshooting

### Issue: `make: Command not found` on Windows
**Solution:** Install GNU Make via Chocolatey or Scoop:
```powershell
choco install make
# or
scoop install make
```

### Issue: Kaggle API authentication fails
**Solution:** Ensure Kaggle credentials are set up:
1. Download `kaggle.json` from Kaggle account settings
2. Place in `~/.kaggle/kaggle.json` (or `C:\Users\<username>\.kaggle\`)

### Issue: Encoding errors with UTF-8 characters
**Solution:** PowerShell should handle UTF-8 automatically. If issues persist, try:
```powershell
[System.Environment]::SetEnvironmentVariable("PYTHONIOENCODING", "utf-8", "User")
```

### Issue: Phase 1 hangs downloading data
**Solution:** Check your internet connection and Kaggle API credentials. Data files are large (~1GB total).

---

## Key Statistics

| Metric | Value |
|--------|-------|
| Kaggle PGN Games | ~23,000 |
| Lichess Games | ~20,000 |
| Combined Dataset | ~43,000 |
| Total Features | 30+ |
| Target Classes | 5 (Elo buckets) or 3 (winner) |
| Pipeline Runtime | ~10-30 minutes (depends on data download) |

---

## Next Steps

After running the pipeline:

1. **Explore Outputs:**
   - `games.csv` — Kaggle games with Stockfish features
   - `merged_games.csv` — Combined dataset for modeling

2. **Prepare for Modeling:**
   - Review `notebooks/` for exploratory analysis
   - Run preprocessing notebooks for train/val/test splits

3. **Train Models:**
   - Execute `src/models/train.py` for baseline models
   - Use MLflow for experiment tracking (configured in `config.toml`)

---

## See Also

- [README.md](README.md) — Project overview
- [makefile](makefile) — Pipeline commands
- [configs/config.toml](configs/config.toml) — Configuration parameters
- [src/data/](src/data/) — Data pipeline source code

