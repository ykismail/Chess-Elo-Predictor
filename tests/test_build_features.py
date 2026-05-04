"""
Unit Tests for Phase 4 — Feature Engineering
==============================================
Tests for feature engineering, encoding, and final dataset integration.

Test Levels:
  • Custom Functions: engineer_features, Elo bucketing, ordinal encoding
  • Feature Shapes: output has correct columns and rows
  • Value Ranges: Elo buckets valid, ordinal encoding 0-4
  • No Missing Data: critical features have no NaN
  • Consistency: target variables properly encoded
  • Reproducibility: same input → same output
  • Validation: features ready for modeling
"""

import os
import sys
import tempfile
import pytest
import pandas as pd
import numpy as np

# ── Path bootstrap ────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, ".."))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# Import from build_features module
sys.path.insert(0, os.path.join(project_dir, "src", "data"))
from build_features import engineer_features, validation_report


class TestEngineerFeatures:
    """Test feature engineering pipeline."""

    def create_sample_kaggle_df(self):
        """Create sample Kaggle merged DataFrame."""
        return pd.DataFrame({
            "event_id": [1, 2, 3, 4, 5],
            "white_elo": [1200, 1500, 2000, 2500, 3000],
            "black_elo": [1300, 1600, 1900, 2400, 2900],
            "white_acl": [80.0, 85.5, 88.3, 90.1, 92.5],
            "black_acl": [75.5, 83.0, 87.1, 89.5, 91.0],
            "result": ["1-0", "0-1", "1/2-1/2", "1-0", "0-1"],
            "num_moves": [20, 35, 42, 58, 45],
            "white_blunders": [2, 1, 0, 0, 1],
            "black_blunders": [1, 2, 1, 1, 0]
        })

    def test_engineer_features_returns_dataframe(self):
        """Test that engineer_features returns a DataFrame."""
        df = self.create_sample_kaggle_df()
        
        df_engineered = engineer_features(df)
        
        assert isinstance(df_engineered, pd.DataFrame), "Should return DataFrame"
        assert len(df_engineered) == len(df), "Should preserve number of rows"

    def test_elo_bucket_columns_created(self):
        """Test that Elo bucket columns are created."""
        df = self.create_sample_kaggle_df()
        
        df_engineered = engineer_features(df)
        
        assert "elo_bucket_white" in df_engineered.columns
        assert "elo_bucket_black" in df_engineered.columns
        assert "elo_bucket_white_categorical" in df_engineered.columns
        assert "elo_bucket_black_categorical" in df_engineered.columns

    def test_elo_bucket_valid_values(self):
        """Test that Elo buckets contain valid skill tier labels."""
        df = self.create_sample_kaggle_df()
        
        df_engineered = engineer_features(df)
        
        valid_buckets = ["Beginner", "Intermediate", "Advanced", "Expert", "Master"]
        assert df_engineered["elo_bucket_white"].isin(valid_buckets).all()
        assert df_engineered["elo_bucket_black"].isin(valid_buckets).all()

    def test_elo_bucket_categorical_ordinal_values(self):
        """Test that categorical Elo buckets are properly ordinal encoded (0-4)."""
        df = self.create_sample_kaggle_df()
        
        df_engineered = engineer_features(df)
        
        # Should be 0-4
        assert df_engineered["elo_bucket_white_categorical"].min() >= 0
        assert df_engineered["elo_bucket_white_categorical"].max() <= 4
        assert df_engineered["elo_bucket_black_categorical"].min() >= 0
        assert df_engineered["elo_bucket_black_categorical"].max() <= 4

    def test_elo_classification_correctness(self):
        """Test that ELO values are correctly classified into buckets."""
        df = pd.DataFrame({
            "event_id": [1, 2, 3, 4, 5],
            "white_elo": [500, 1200, 1700, 2200, 2900],
            "black_elo": [500, 1200, 1700, 2200, 2900],
            "white_acl": [80.0] * 5,
            "black_acl": [80.0] * 5,
            "result": ["1-0"] * 5,
            "num_moves": [20] * 5,
            "white_blunders": [0] * 5,
            "black_blunders": [0] * 5
        })
        
        df_engineered = engineer_features(df)
        
        # Check bucket assignments
        assert df_engineered.iloc[0]["elo_bucket_white"] == "Beginner"  # 500
        assert df_engineered.iloc[1]["elo_bucket_white"] == "Beginner"  # 1200
        assert df_engineered.iloc[2]["elo_bucket_white"] == "Intermediate"  # 1700
        assert df_engineered.iloc[3]["elo_bucket_white"] == "Advanced"  # 2200
        assert df_engineered.iloc[4]["elo_bucket_white"] == "Expert"  # 2900

    def test_acl_gap_calculation(self):
        """Test that ACL gap is correctly calculated."""
        df = pd.DataFrame({
            "event_id": [1, 2],
            "white_elo": [1600, 1400],
            "black_elo": [1500, 1600],
            "white_acl": [85.0, 90.0],
            "black_acl": [80.0, 85.0],
            "result": ["1-0", "0-1"],
            "num_moves": [20, 30],
            "white_blunders": [0, 1],
            "black_blunders": [1, 0]
        })
        
        df_engineered = engineer_features(df)
        
        # Check acl_gap calculation
        assert df_engineered.iloc[0]["acl_gap"] == pytest.approx(5.0, abs=0.1)
        assert df_engineered.iloc[1]["acl_gap"] == pytest.approx(5.0, abs=0.1)

    def test_winner_binary_encoding(self):
        """Test that winner_binary is correctly encoded (1=White, 0=Black)."""
        df = pd.DataFrame({
            "event_id": [1, 2, 3],
            "white_elo": [1600, 1600, 1600],
            "black_elo": [1500, 1500, 1500],
            "white_acl": [85.0, 85.0, 85.0],
            "black_acl": [80.0, 80.0, 80.0],
            "result": ["1-0", "0-1", "1/2-1/2"],
            "num_moves": [20, 30, 40],
            "white_blunders": [0, 1, 0],
            "black_blunders": [1, 0, 0]
        })
        
        df_engineered = engineer_features(df)
        
        assert df_engineered.iloc[0]["winner_binary"] == 1  # White wins
        assert df_engineered.iloc[1]["winner_binary"] == 0  # Black wins
        assert pd.isna(df_engineered.iloc[2]["winner_binary"])  # Draw -> NaN

    def test_winner_multiclass_encoding(self):
        """Test that winner_multiclass is correctly encoded (0=Black, 1=Draw, 2=White)."""
        df = pd.DataFrame({
            "event_id": [1, 2, 3],
            "white_elo": [1600, 1600, 1600],
            "black_elo": [1500, 1500, 1500],
            "white_acl": [85.0, 85.0, 85.0],
            "black_acl": [80.0, 80.0, 80.0],
            "result": ["1-0", "0-1", "1/2-1/2"],
            "num_moves": [20, 30, 40],
            "white_blunders": [0, 1, 0],
            "black_blunders": [1, 0, 0]
        })
        
        df_engineered = engineer_features(df)
        
        assert df_engineered.iloc[0]["winner_multiclass"] == 2  # White wins
        assert df_engineered.iloc[1]["winner_multiclass"] == 0  # Black wins
        assert df_engineered.iloc[2]["winner_multiclass"] == 1  # Draw

    def test_no_missing_engineered_features(self):
        """Test that critical engineered features have no NaN (except draws in binary)."""
        df = self.create_sample_kaggle_df()
        
        df_engineered = engineer_features(df)
        
        # These should have no NaN
        critical_cols = ["elo_bucket_white_categorical", "acl_gap", "winner_multiclass"]
        for col in critical_cols:
            assert df_engineered[col].isna().sum() == 0, f"{col} should not have NaN"

    def test_engineer_features_reproducibility(self):
        """Test that feature engineering is reproducible for same input."""
        df = self.create_sample_kaggle_df()
        
        df_eng1 = engineer_features(df)
        df_eng2 = engineer_features(df)
        
        pd.testing.assert_frame_equal(df_eng1, df_eng2, check_dtype=True)


class TestFeatureDataTypes:
    """Test that engineered features have correct data types."""

    def create_sample_df(self):
        """Create sample DataFrame."""
        return pd.DataFrame({
            "event_id": [1, 2, 3],
            "white_elo": [1600, 1400, 1800],
            "black_elo": [1500, 1600, 1700],
            "white_acl": [85.5, 90.0, 88.3],
            "black_acl": [80.0, 92.5, 87.1],
            "result": ["1-0", "0-1", "1/2-1/2"],
            "num_moves": [20, 35, 42],
            "white_blunders": [0, 1, 0],
            "black_blunders": [1, 0, 0]
        })

    def test_categorical_columns_object_type(self):
        """Test that categorical columns are object dtype."""
        df = self.create_sample_df()
        df_eng = engineer_features(df)
        
        assert df_eng["elo_bucket_white"].dtype == "object"
        assert df_eng["elo_bucket_black"].dtype == "object"

    def test_ordinal_columns_numeric_type(self):
        """Test that ordinal columns are numeric dtype."""
        df = self.create_sample_df()
        df_eng = engineer_features(df)
        
        assert pd.api.types.is_numeric_dtype(df_eng["elo_bucket_white_categorical"])
        assert pd.api.types.is_numeric_dtype(df_eng["elo_bucket_black_categorical"])
        assert pd.api.types.is_numeric_dtype(df_eng["acl_gap"])


class TestFeatureConsistency:
    """Test consistency and integrity of engineered features."""

    def create_sample_df(self):
        """Create sample DataFrame."""
        return pd.DataFrame({
            "event_id": [1, 2, 3],
            "white_elo": [1600, 1400, 1800],
            "black_elo": [1500, 1600, 1700],
            "white_acl": [85.5, 90.0, 88.3],
            "black_acl": [80.0, 92.5, 87.1],
            "result": ["1-0", "0-1", "1/2-1/2"],
            "num_moves": [20, 35, 42],
            "white_blunders": [0, 1, 0],
            "black_blunders": [1, 0, 0]
        })

    def test_no_duplicate_rows_after_engineering(self):
        """Test that feature engineering does not create duplicates."""
        df = self.create_sample_df()
        df_eng = engineer_features(df)
        
        assert len(df_eng) == len(df), "Should have same number of rows"
        assert df_eng.duplicated(subset=["event_id"]).sum() == 0, "No duplicate event_ids"

    def test_event_id_preserved(self):
        """Test that event_id is preserved through engineering."""
        df = self.create_sample_df()
        df_eng = engineer_features(df)
        
        assert (df_eng["event_id"] == df["event_id"]).all(), "event_id should be preserved"

    def test_original_columns_preserved(self):
        """Test that original columns are preserved in engineered output."""
        df = self.create_sample_df()
        df_eng = engineer_features(df)
        
        original_cols = ["event_id", "white_elo", "black_elo", "result"]
        for col in original_cols:
            assert col in df_eng.columns, f"Original column '{col}' should be preserved"


class TestValidationReport:
    """Test validation report generation."""

    def test_validation_report_no_error(self, capsys):
        """Test that validation_report runs without error."""
        df = pd.DataFrame({
            "event_id": [1, 2, 3],
            "white_elo": [1600, 1400, 1800],
            "black_elo": [1500, 1600, 1700],
            "white_acl": [85.5, 90.0, 88.3],
            "elo_bucket_white": ["Intermediate", "Beginner", "Advanced"],
            "elo_bucket_white_categorical": [1, 0, 2]
        })
        
        # Should not raise exception
        validation_report(df)
        
        captured = capsys.readouterr()
        assert "VALIDATION REPORT" in captured.out

    def test_validation_report_handles_missing_columns(self, capsys):
        """Test that validation_report handles DataFrames with missing columns gracefully."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        })
        
        # Should not raise exception
        validation_report(df)
        
        captured = capsys.readouterr()
        assert "Shape:" in captured.out


class TestFeatureIntegration:
    """Test integration of all features for modeling readiness."""

    def test_engineered_df_ready_for_modeling(self):
        """Test that engineered DataFrame has all required features for modeling."""
        df = pd.DataFrame({
            "event_id": [1, 2, 3],
            "white_elo": [1600, 1400, 1800],
            "black_elo": [1500, 1600, 1700],
            "white_acl": [85.5, 90.0, 88.3],
            "black_acl": [80.0, 92.5, 87.1],
            "result": ["1-0", "0-1", "1/2-1/2"],
            "num_moves": [20, 35, 42],
            "white_blunders": [0, 1, 0],
            "black_blunders": [1, 0, 0]
        })
        
        df_eng = engineer_features(df)
        
        # Check for target variables
        assert "winner_binary" in df_eng.columns or "winner_multiclass" in df_eng.columns
        
        # Check for feature variables
        assert "elo_bucket_white_categorical" in df_eng.columns
        assert "acl_gap" in df_eng.columns

    def test_no_infinite_values_in_features(self):
        """Test that engineered features have no infinite values."""
        df = pd.DataFrame({
            "event_id": [1, 2, 3],
            "white_elo": [1600, 1400, 1800],
            "black_elo": [1500, 1600, 1700],
            "white_acl": [85.5, 90.0, 88.3],
            "black_acl": [80.0, 92.5, 87.1],
            "result": ["1-0", "0-1", "1/2-1/2"],
            "num_moves": [20, 35, 42],
            "white_blunders": [0, 1, 0],
            "black_blunders": [1, 0, 0]
        })
        
        df_eng = engineer_features(df)
        
        numeric_cols = df_eng.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert not np.isinf(df_eng[col]).any(), f"Column '{col}' should not have infinite values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
