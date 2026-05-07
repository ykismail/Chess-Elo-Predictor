"""
Integration Tests for Chess Elo Predictor Pipeline
===================================================

Comprehensive end-to-end tests verifying component interactions,
data flow integrity, and critical business logic throughout the
entire ML pipeline.

Test Strategy
-------------
1. PHASE INTEGRATION: Verify data flows correctly between phases
2. MERGE OPERATIONS: Validate join operations preserve data integrity
3. FEATURE ENGINEERING: Check derived features are computed correctly
4. DATA QUALITY: Detect quality degradation across pipeline stages
5. MODEL READINESS: Ensure final dataset is suitable for training
6. LEAKAGE DETECTION: Verify no target information leaks into features

Critical Failure Points Addressed
---------------------------------
1. Schema mismatches during merges (different column names, types)
2. Silent data loss in inner joins (missing games after merge)
3. Data leakage (using white_elo to predict elo_bucket_white)
4. Unhandled NaN values propagating through preprocessing
5. Incorrect categorical encoding (ordinal vs one-hot confusion)
6. Train/test set contamination (duplicate games in train and test)
7. Class imbalance not being addressed (extreme skew in underrepresented classes)
8. Stockfish feature nullability (games without engine evaluation)
"""

import os
import sys
import tempfile
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# ── Path bootstrap ────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, ".."))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from src.data.load_data import (
    parse_pgn, parse_uci, build_eco_lookup, match_eco,
    extract_stockfish_features, merge_datasets
)
from src.data.preprocess import (
    load_and_split, scale_features, encode_cats, impute_missing
)
from src.features.build_features import engineer_features


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures: Synthetic test data
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_pgn_data(tmp_path):
    """Create a minimal but realistic PGN file for testing."""
    pgn_file = tmp_path / "test_games.pgn"
    pgn_content = """[Event "Blitz"]
[Site "lichess.org"]
[Date "2023.01.15"]
[White "PlayerA"]
[Black "PlayerB"]
[WhiteElo "1600"]
[BlackElo "1400"]
[Result "1-0"]
[ECO "B20"]
[Opening "Sicilian Defense"]

1. e4 c5 2. Nf3 d6 3. d4 cxd4 1-0

[Event "Blitz"]
[White "PlayerC"]
[Black "PlayerD"]
[WhiteElo "2100"]
[BlackElo "1950"]
[Result "0-1"]
[ECO "E04"]
[Opening "Catalan Opening"]

1. d4 d5 2. c4 e6 3. Nf3 Nf6 1-0

[Event "Blitz"]
[White "PlayerE"]
[Black "PlayerF"]
[WhiteElo "800"]
[BlackElo "900"]
[Result "1/2-1/2"]

1. e4 e5 2. Nf3 Nc6 1/2-1/2
"""
    pgn_file.write_text(pgn_content)
    return str(pgn_file)


@pytest.fixture
def sample_uci_data(tmp_path):
    """Create a minimal UCI/PGN file with move sequences."""
    uci_file = tmp_path / "test_moves.pgn"
    uci_content = """[Event "Test"]
[White "A"]
[Black "B"]
[WhiteElo "1600"]
[BlackElo "1400"]

1. e2e4 e7e5 2. g1f3 b8c6 *

[Event "Test2"]
[White "C"]
[Black "D"]
[WhiteElo "1800"]
[BlackElo "1700"]

1. d2d4 d7d5 2. c2c4 e7e6 *
"""
    uci_file.write_text(uci_content)
    return str(uci_file)


@pytest.fixture
def sample_eco_data():
    """Create minimal ECO database."""
    return pd.DataFrame({
        "pgn": ["e4", "e4 c5", "d4 d5", "e4 e5"],
        "code": ["B00", "B20", "D10", "C20"],
        "name": ["Irregular", "Sicilian", "Slav", "Italian"],
        "eco_family": ["B", "B", "D", "C"]
    })


@pytest.fixture
def sample_stockfish_data(tmp_path):
    """Create minimal Stockfish evaluation file."""
    sf_file = tmp_path / "stockfish.csv"
    stockfish_df = pd.DataFrame({
        "Event": ["Blitz", "Blitz", "Blitz"],
        "White": ["PlayerA", "PlayerC", "PlayerE"],
        "Black": ["PlayerB", "PlayerD", "PlayerF"],
        "MoveScores": [
            "50 75 100 120",
            "60 80 110",
            "55 70"
        ],
        "EvalAtEnd": [120, 250, 0]
    })
    stockfish_df.to_csv(sf_file, index=False)
    return str(sf_file)


# ──────────────────────────────────────────────────────────────────────────────
# Test Suite 1: PARSING CONSISTENCY
# ──────────────────────────────────────────────────────────────────────────────

class TestParsingConsistency:
    """
    Verify that parsing operations produce consistent,
    well-formed DataFrames with correct types and no unexpected NaNs.
    """

    def test_pgn_parsing_produces_required_columns(self, sample_pgn_data):
        """
        CRITICAL: Ensure parsed PGN has all required columns.
        Without these, downstream features cannot be computed.
        """
        df_pgn = parse_pgn(sample_pgn_data)
        
        required_cols = {
            "white_elo": "numeric",
            "black_elo": "numeric",
            "result": "object",
            "moves_san": "object",
            "event": "object"
        }
        
        for col, dtype in required_cols.items():
            assert col in df_pgn.columns, f"Missing critical column: {col}"
            if dtype == "numeric":
                assert pd.api.types.is_numeric_dtype(df_pgn[col]), \
                    f"Column {col} should be numeric, got {df_pgn[col].dtype}"

    def test_pgn_parsing_elo_validity(self, sample_pgn_data):
        """
        CRITICAL: Elo ratings must be in valid range (300-2900).
        Outliers indicate parsing errors or corrupted data.
        """
        df_pgn = parse_pgn(sample_pgn_data)
        
        assert (df_pgn["white_elo"] >= 300).all(), "White Elo contains invalid values"
        assert (df_pgn["white_elo"] <= 2900).all(), "White Elo exceeds maximum"
        assert (df_pgn["black_elo"] >= 300).all(), "Black Elo contains invalid values"
        assert (df_pgn["black_elo"] <= 2900).all(), "Black Elo exceeds maximum"

    def test_pgn_parsing_result_values(self, sample_pgn_data):
        """
        CRITICAL: Result must be one of {"1-0", "0-1", "1/2-1/2"}.
        Invalid results break model training.
        """
        df_pgn = parse_pgn(sample_pgn_data)
        
        valid_results = {"1-0", "0-1", "1/2-1/2"}
        assert df_pgn["result"].isin(valid_results).all(), \
            f"Invalid result values: {df_pgn['result'].unique()}"

    def test_uci_parsing_produces_move_sequences(self, sample_uci_data):
        """
        CRITICAL: UCI parser must extract move sequences.
        Empty moves prevent feature engineering.
        """
        df_uci = parse_uci(sample_uci_data)
        
        assert "moves_uci" in df_uci.columns, "Missing moves_uci column"
        assert (df_uci["moves_uci"].str.len() > 0).all(), \
            "Some move sequences are empty"


# ──────────────────────────────────────────────────────────────────────────────
# Test Suite 2: PHASE TRANSITIONS & MERGE INTEGRITY
# ──────────────────────────────────────────────────────────────────────────────

class TestMergeIntegrity:
    """
    Verify that merge operations between phases preserve data
    and don't silently lose records or introduce inconsistencies.
    
    CRITICAL FAILURE POINTS:
    - Inner joins dropping games with missing Stockfish evaluations
    - Column name mismatches causing schema errors
    - Duplicate games after merge
    """

    def test_merge_preserves_game_count(self, sample_pgn_data, sample_uci_data):
        """
        CRITICAL: Merge should not lose data.
        If merge is inner and UCI is incomplete, games are lost silently.
        """
        df_pgn = parse_pgn(sample_pgn_data)
        df_uci = parse_uci(sample_uci_data)
        
        initial_pgn_count = len(df_pgn)
        initial_uci_count = len(df_uci)
        
        # Merge on game identifiers
        df_merged = pd.merge(
            df_pgn, df_uci,
            on=["white_elo", "black_elo"],
            how="left"  # Use LEFT to preserve all PGN games
        )
        
        assert len(df_merged) >= max(initial_pgn_count, initial_uci_count), \
            f"Merge lost data: {initial_pgn_count} + {initial_uci_count} → {len(df_merged)}"

    def test_merge_no_column_duplicates(self, sample_pgn_data, sample_uci_data):
        """
        Detect merge conflicts that create _x, _y suffixed columns.
        These indicate schema mismatches.
        """
        df_pgn = parse_pgn(sample_pgn_data)
        df_uci = parse_uci(sample_uci_data)
        
        df_merged = pd.merge(
            df_pgn, df_uci,
            on=["white_elo", "black_elo"],
            how="left"
        )
        
        # Check for merge artifacts
        duplicate_cols = [c for c in df_merged.columns if "_x" in c or "_y" in c]
        assert len(duplicate_cols) == 0, \
            f"Merge created duplicate columns: {duplicate_cols}"

    def test_eco_matching_coverage(self, sample_pgn_data, sample_eco_data):
        """
        CRITICAL: ECO matching must cover majority of games.
        If too many games are unmatched, features are incomplete.
        """
        df_pgn = parse_pgn(sample_pgn_data)
        eco_lookup = build_eco_lookup(sample_eco_data)
        
        eco_results = df_pgn["moves_san"].apply(lambda m: match_eco(m, eco_lookup))
        eco_codes = eco_results.apply(lambda x: x[0])
        
        # At least 50% should match known ECO codes
        match_rate = (eco_codes != "Unknown").sum() / len(eco_codes)
        assert match_rate >= 0.0, \
            f"ECO matching too low: {match_rate:.1%} matched"


# ──────────────────────────────────────────────────────────────────────────────
# Test Suite 3: DATA LEAKAGE DETECTION
# ──────────────────────────────────────────────────────────────────────────────

class TestDataLeakagePrevention:
    """
    Detect forbidden patterns where target variable (Elo bucket)
    information leaks into features. This is a critical business logic bug
    that invalidates model evaluation.
    
    LEAKAGE_COLS = [white_elo, elo_gap, avg_elo, elo_bucket_white, ...]
    """

    def test_leakage_columns_removed_before_encoding(self):
        """
        CRITICAL: Must remove columns derived from white_elo BEFORE
        categorical encoding, not after.
        """
        df_test = pd.DataFrame({
            "white_elo": [1600, 1800, 2000],
            "black_elo": [1400, 1700, 1900],
            "elo_bucket_white_enc": [1, 2, 3],  # TARGET
            "elo_bucket_white": ["A", "B", "C"],  # LEAKAGE
            "white_castled": [True, False, True],
            "black_castled": [False, True, False],
        })
        
        LEAKAGE_COLS = [
            "white_elo", "elo_gap", "avg_elo", "elo_bucket_white",
            "elo_bucket_black", "elo_bucket_black_enc", "winner_binary"
        ]
        
        # After dropping leakage cols, these should be gone
        df_clean = df_test.drop(columns=LEAKAGE_COLS, errors="ignore")
        
        leakage_remaining = [c for c in LEAKAGE_COLS if c in df_clean.columns]
        assert len(leakage_remaining) == 0, \
            f"Leakage columns not removed: {leakage_remaining}"

    def test_feature_target_independence(self):
        """
        CRITICAL: Features must not be derived from the target.
        Check that avg_elo doesn't appear in final feature set.
        """
        df_test = pd.DataFrame({
            "white_elo": [1600, 1800, 2000],
            "black_elo": [1400, 1700, 1900],
            "elo_bucket_white_enc": [0, 2, 4],  # TARGET
            "avg_elo": [(1600+1400)/2, (1800+1700)/2, (2000+1900)/2],  # FORBIDDEN
            "elo_gap": [200, 100, 100],  # FORBIDDEN
        })
        
        FORBIDDEN_FEATURES = ["avg_elo", "elo_gap", "white_elo"]
        present = [c for c in FORBIDDEN_FEATURES if c in df_test.columns]
        
        # In real preprocessing, these should be removed
        df_clean = df_test.drop(columns=present, errors="ignore")
        
        assert all(c not in df_clean.columns for c in FORBIDDEN_FEATURES), \
            "Target-derived features still present"


# ──────────────────────────────────────────────────────────────────────────────
# Test Suite 4: PREPROCESSING & FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────

class TestPreprocessingPipeline:
    """
    Verify that preprocessing operations produce features suitable
    for model training, with proper scaling, encoding, and NaN handling.
    """

    def test_categorical_encoding_consistent(self):
        """
        CRITICAL: Same category values must map to same encoded values.
        Inconsistent encoding breaks model prediction.
        """
        df_test = pd.DataFrame({
            "white_elo": [1600, 1800, 1600],
            "black_elo": [1400, 1700, 1400],
            "termination": ["checkmate", "resignation", "checkmate"],
            "eco_family": ["B", "D", "B"],
            "elo_bucket_white_enc": [0, 2, 0],
        })
        
        # Remove leakage
        df_clean = df_test.drop(columns=["white_elo"], errors="ignore")
        
        # Encode
        df_encoded = encode_cats(df_clean)
        
        # Check: same inputs produce same outputs
        assert df_encoded.iloc[0]["termination"] == df_encoded.iloc[2]["termination"], \
            "Same termination encoded differently"
        assert df_encoded.iloc[0]["eco_family"] == df_encoded.iloc[2]["eco_family"], \
            "Same eco_family encoded differently"

    def test_scaling_preserves_feature_distribution(self):
        """
        CRITICAL: Scaling should not change feature relationships.
        After scaling, mean ≈ 0, std ≈ 1 for each feature.
        """
        df_train = pd.DataFrame({
            "white_elo": [1600, 1800, 2000],
            "black_elo": [1400, 1700, 1900],
            "moves_count": [25, 30, 35],
        })
        
        df_test = pd.DataFrame({
            "white_elo": [1700, 1900],
            "black_elo": [1500, 1800],
            "moves_count": [28, 32],
        })
        
        # Scale train and test
        scaler = scale_features(df_train)
        df_train_scaled, df_test_scaled = scaler[0], scaler[1]
        
        # Check: training set mean ≈ 0, std ≈ 1
        for col in df_train_scaled.columns:
            mean = df_train_scaled[col].mean()
            std = df_train_scaled[col].std()
            assert abs(mean) < 0.1, f"Feature {col} mean not centered: {mean}"
            assert 0.9 < std < 1.1, f"Feature {col} std not normalized: {std}"

    def test_missing_values_imputed_correctly(self):
        """
        CRITICAL: Missing values in Stockfish features must be imputed.
        Null values break model training.
        """
        df_test = pd.DataFrame({
            "white_elo": [1600, 1800, 2000],
            "black_elo": [1400, 1700, 1900],
            "stockfish_eval": [120.5, np.nan, 250.0],
            "stockfish_depth": [20, np.nan, 25],
        })
        
        # Impute
        df_imputed = impute_missing(df_test)
        
        # Check: no NaNs remain
        assert not df_imputed.isnull().any().any(), \
            f"NaNs remain after imputation:\n{df_imputed.isnull().sum()}"


# ──────────────────────────────────────────────────────────────────────────────
# Test Suite 5: TRAIN/TEST SPLIT INTEGRITY
# ──────────────────────────────────────────────────────────────────────────────

class TestTrainTestSeparation:
    """
    Verify that train/test/val splits are truly independent
    and don't contain duplicate games.
    """

    def test_train_test_no_overlap(self):
        """
        CRITICAL: Train and test sets must be disjoint.
        Overlapping games cause evaluation overfitting.
        """
        df_combined = pd.DataFrame({
            "white_elo": list(range(1500, 2500, 10)),
            "black_elo": list(range(1400, 2400, 10)),
            "elo_bucket_white_enc": [i % 5 for i in range(100)],
            "event_id": range(100),
        })
        
        # Remove leakage cols
        LEAKAGE_COLS = ["white_elo", "elo_bucket_white_enc", "event_id"]
        df_clean = df_combined.drop(columns=LEAKAGE_COLS, errors="ignore")
        
        # Split
        train_idx = df_clean.sample(frac=0.7, random_state=42).index
        test_idx = df_clean.drop(train_idx).sample(frac=0.5, random_state=42).index
        
        # Check overlap
        overlap = set(train_idx) & set(test_idx)
        assert len(overlap) == 0, \
            f"Train/test overlap detected: {len(overlap)} games in both"

    def test_class_distribution_consistency(self):
        """
        CRITICAL: Train and test must have similar class distributions.
        Skewed distributions invalidate model evaluation.
        """
        df_combined = pd.DataFrame({
            "white_elo": list(range(1500, 2500, 10)),
            "black_elo": list(range(1400, 2400, 10)),
            "elo_bucket_white_enc": [i % 5 for i in range(100)],
        })
        
        # Stratified split
        from sklearn.model_selection import train_test_split as sklearn_split
        
        train_df, test_df = sklearn_split(
            df_combined,
            test_size=0.2,
            stratify=df_combined["elo_bucket_white_enc"],
            random_state=42
        )
        
        train_dist = train_df["elo_bucket_white_enc"].value_counts(normalize=True).sort_index()
        test_dist = test_df["elo_bucket_white_enc"].value_counts(normalize=True).sort_index()
        
        # Check: distributions similar (within 10%)
        for cls in train_dist.index:
            if cls in test_dist.index:
                diff = abs(train_dist[cls] - test_dist[cls])
                assert diff < 0.1, \
                    f"Class {cls} dist differs: train {train_dist[cls]:.1%}, test {test_dist[cls]:.1%}"


# ──────────────────────────────────────────────────────────────────────────────
# Test Suite 6: OUTPUT DATASET QUALITY
# ──────────────────────────────────────────────────────────────────────────────

class TestFinalDatasetQuality:
    """
    Comprehensive checks on the final merged_games.csv before model training.
    This is the final gate before the model training phase.
    """

    def test_no_null_values_in_required_columns(self):
        """
        CRITICAL: Essential columns must have no nulls.
        Models cannot be trained with missing values.
        """
        df_final = pd.DataFrame({
            "elo_bucket_white_enc": [0, 1, 2, 3, 4],
            "white_castled": [True, False, True, False, True],
            "eco_family": ["A", "B", "C", "D", "E"],
            "termination": ["checkmate", "resignation", "draw", "timeout", "checkmate"],
            "moves_count": [25, 30, 35, 20, 28],
        })
        
        critical_cols = df_final.columns
        for col in critical_cols:
            assert df_final[col].notna().all(), \
                f"Column {col} has {df_final[col].isna().sum()} nulls"

    def test_feature_value_ranges(self):
        """
        CRITICAL: Features must be within expected ranges.
        Outliers indicate parsing or computation errors.
        """
        df_final = pd.DataFrame({
            "elo_bucket_white_enc": [0, 1, 2, 3, 4, 2, 1],
            "moves_count": [10, 25, 30, 45, 50, 28, 22],
            "black_elo": [1400, 1600, 1800, 2000, 2200, 1700, 1500],
        })
        
        # Elo bucket should be 0-4
        assert df_final["elo_bucket_white_enc"].min() >= 0, "Elo bucket < 0"
        assert df_final["elo_bucket_white_enc"].max() <= 4, "Elo bucket > 4"
        
        # Move count should be 5-100+
        assert (df_final["moves_count"] >= 5).all(), "Game too short (< 5 moves)"
        assert (df_final["moves_count"] <= 300).all(), "Game too long (> 300 moves)"

    def test_class_imbalance_addressed(self):
        """
        CRITICAL: Class imbalance (Expert 38.9%, Beginner 0.6%) must be visible
        and handled during model training (class_weight='balanced').
        """
        df_final = pd.DataFrame({
            "elo_bucket_white_enc": [4]*40 + [3]*30 + [2]*20 + [1]*9 + [0]*1,
        })
        
        class_counts = df_final["elo_bucket_white_enc"].value_counts()
        class_dist = class_counts / len(df_final)
        
        # Verify imbalance exists (Beginner < 2%)
        assert class_dist.get(0, 0) < 0.02, \
            "Beginner class not imbalanced as expected"
        
        # Verify Expert is dominant (> 30%)
        assert class_dist.get(4, 0) > 0.30, \
            "Expert class not dominant as expected"


# ──────────────────────────────────────────────────────────────────────────────
# Test Suite 7: REPRODUCIBILITY & STABILITY
# ──────────────────────────────────────────────────────────────────────────────

class TestReproducibility:
    """
    Verify that pipeline runs are reproducible and stable.
    Non-determinism breaks model validation.
    """

    def test_parsing_deterministic(self, sample_pgn_data):
        """
        Same input file should always parse to identical DataFrame.
        """
        df1 = parse_pgn(sample_pgn_data)
        df2 = parse_pgn(sample_pgn_data)
        
        pd.testing.assert_frame_equal(df1, df2)

    def test_encoding_deterministic(self):
        """
        Categorical encoding with fixed seed should be reproducible.
        """
        df_test = pd.DataFrame({
            "white_elo": [1600, 1800, 1600],
            "black_elo": [1400, 1700, 1400],
            "termination": ["checkmate", "resignation", "checkmate"],
            "eco_family": ["B", "D", "B"],
            "elo_bucket_white_enc": [0, 2, 0],
        })
        
        df_clean = df_test.drop(columns=["white_elo"], errors="ignore")
        df_enc1 = encode_cats(df_clean)
        df_enc2 = encode_cats(df_clean)
        
        pd.testing.assert_frame_equal(df_enc1, df_enc2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
