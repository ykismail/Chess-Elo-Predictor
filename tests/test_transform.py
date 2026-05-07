"""
Unit Tests for Phase 3 — Transformation & Encoding
===================================================
Tests for ECO matching, Stockfish extraction, and merge operations.

Test Levels:
  • Custom Functions: ECO matching, merge operations
  • Data Shapes: output rows/columns match expectations
  • No Data Loss: joins preserve all expected records
  • Reproducibility: same input → same output
  • Value Ranges: features within expected bounds
  • Consistency: no unexpected NaN, duplicates
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

from src.data.load_data import (
    build_eco_lookup,
    match_eco,
    merge_datasets,
    extract_stockfish_features,
)


class TestECOMatching:
    """Test ECO opening code matching functionality."""

    def test_build_eco_lookup_returns_dict(self):
        """Test that build_eco_lookup returns a dictionary."""
        df_eco = pd.DataFrame({
            "pgn": ["e2 e4", "e2 e4 c7 c5"],
            "code": ["B00", "B20"],
            "name": ["Irregular Opening", "Sicilian Defense"]
        })
        
        lookup = build_eco_lookup(df_eco)
        
        assert isinstance(lookup, dict), "build_eco_lookup should return dict"
        assert len(lookup) > 0, "Lookup should have entries"

    def test_match_eco_returns_tuple(self):
        """Test that match_eco returns a tuple of (code, name, family)."""
        df_eco = pd.DataFrame({
            "pgn": ["e4", "e4 c5"],
            "code": ["B00", "B20"],
            "name": ["Irregular", "Sicilian"]
        })
        lookup = build_eco_lookup(df_eco)
        
        result = match_eco("e4 c5 Nf3", lookup)
        
        assert isinstance(result, tuple), "match_eco should return tuple"
        assert len(result) == 3, "Tuple should have 3 elements (code, name, family)"

    def test_match_eco_handles_no_match(self):
        """Test that match_eco returns 'Unknown' for unmatched moves."""
        df_eco = pd.DataFrame({
            "pgn": ["e4"],
            "code": ["B00"],
            "name": ["Irregular"]
        })
        lookup = build_eco_lookup(df_eco)
        
        result = match_eco("a1a2", lookup)  # Very unusual move sequence
        
        assert result[0] == "Unknown", "Unmatched sequences should return Unknown"

    def test_eco_lookup_reproducibility(self):
        """Test that eco matching is reproducible for same input."""
        df_eco = pd.DataFrame({
            "pgn": ["e4 c5", "d4 d5"],
            "code": ["B20", "D10"],
            "name": ["Sicilian", "Slav"]
        })
        lookup = build_eco_lookup(df_eco)
        
        result1 = match_eco("e4 c5 Nf3", lookup)
        result2 = match_eco("e4 c5 Nf3", lookup)
        
        assert result1 == result2, "Same move sequence should produce same ECO"


class TestStockfishExtraction:
    """Test Stockfish feature extraction functionality."""

    def test_extract_stockfish_returns_dataframe(self, tmp_path):
        """Test that extract_stockfish_features returns a DataFrame."""
        sf_file = tmp_path / "stockfish.csv"
        sf_content = """Event,MoveScores,EvalAtEnd
1,"[100, 50, -30]",0.5
2,"[50, -50, 0]",-0.3
"""
        sf_file.write_text(sf_content)
        
        df = extract_stockfish_features(str(sf_file))
        
        assert isinstance(df, pd.DataFrame), "Should return DataFrame"
        assert len(df) > 0, "DataFrame should have rows"

    def test_stockfish_output_columns(self, tmp_path):
        """Test that extracted Stockfish has required columns."""
        sf_file = tmp_path / "stockfish.csv"
        sf_content = """Event,MoveScores,EvalAtEnd
1,"[100, 50, -30]",0.5
2,"[50, -50, 0]",-0.3
"""
        sf_file.write_text(sf_content)
        
        df = extract_stockfish_features(str(sf_file))
        
        required_cols = ["event_id", "white_acl", "black_acl"]
        for col in required_cols:
            assert col in df.columns, f"Column '{col}' not found in Stockfish features"

    def test_stockfish_numeric_outputs(self, tmp_path):
        """Test that Stockfish output features are numeric."""
        sf_file = tmp_path / "stockfish.csv"
        sf_content = """Event,MoveScores,EvalAtEnd
1,"[100, 50, -30]",0.5
2,"[50, -50, 0]",-0.3
"""
        sf_file.write_text(sf_content)
        
        df = extract_stockfish_features(str(sf_file))
        
        numeric_cols = ["white_acl", "black_acl"]
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} should be numeric"


class TestMergeOperations:
    """Test merge/join operations between different dataframes."""

    def create_sample_pgn_df(self):
        """Create sample PGN DataFrame for testing."""
        return pd.DataFrame({
            "event_id": [1, 2, 3],
            "white_elo": [1600, 1400, 1800],
            "black_elo": [1500, 1600, 1700],
            "result": ["1-0", "0-1", "1/2-1/2"],
            "moves_san": ["e4 c5", "d4 d5", "Nf3 Nf6"]
        })

    def create_sample_uci_df(self):
        """Create sample UCI DataFrame for testing."""
        return pd.DataFrame({
            "event_id": [1, 2, 3],
            "moves_uci": ["e2e4 c7c5", "d2d4 d7d5", "g1f3 g8f6"]
        })

    def create_sample_stockfish_df(self):
        """Create sample Stockfish DataFrame for testing."""
        return pd.DataFrame({
            "event_id": [1, 2, 3],
            "white_acl": [85.5, 90.0, 88.3],
            "black_acl": [80.0, 92.5, 87.1],
            "white_blunders": [0, 1, 0],
            "black_blunders": [1, 0, 0]
        })

    def test_merge_preserves_all_records(self):
        """Test that merge does not lose records (inner join with all present)."""
        df_pgn = self.create_sample_pgn_df()
        df_uci = self.create_sample_uci_df()
        df_sf = self.create_sample_stockfish_df()
        
        df_merged = merge_datasets(df_pgn, df_uci, df_sf)
        
        assert len(df_merged) == 3, "Merge should preserve all records"

    def test_merge_output_shape(self):
        """Test that merge output has expected columns from all inputs."""
        df_pgn = self.create_sample_pgn_df()
        df_uci = self.create_sample_uci_df()
        df_sf = self.create_sample_stockfish_df()
        
        df_merged = merge_datasets(df_pgn, df_uci, df_sf)
        
        # Check columns from each source are present
        assert "white_elo" in df_merged.columns, "Should have PGN columns"
        assert "moves_uci" in df_merged.columns, "Should have UCI columns"
        assert "white_acl" in df_merged.columns, "Should have Stockfish columns"

    def test_merge_no_unexpected_nulls(self):
        """Test that merge does not introduce unexpected NaN values."""
        df_pgn = self.create_sample_pgn_df()
        df_uci = self.create_sample_uci_df()
        df_sf = self.create_sample_stockfish_df()
        
        df_merged = merge_datasets(df_pgn, df_uci, df_sf)
        
        # Key columns should not have NaN
        key_cols = ["event_id", "white_elo", "moves_uci", "white_acl"]
        for col in key_cols:
            assert df_merged[col].isna().sum() == 0, f"Column '{col}' should not have NaN"

    def test_merge_no_duplicates(self):
        """Test that merge does not create duplicate rows."""
        df_pgn = self.create_sample_pgn_df()
        df_uci = self.create_sample_uci_df()
        df_sf = self.create_sample_stockfish_df()
        
        df_merged = merge_datasets(df_pgn, df_uci, df_sf)
        
        # event_id should be unique after merge
        assert df_merged.duplicated(subset=["event_id"]).sum() == 0, "No duplicate event_ids"

    def test_merge_reproducibility(self):
        """Test that merge is reproducible for same input."""
        df_pgn = self.create_sample_pgn_df()
        df_uci = self.create_sample_uci_df()
        df_sf = self.create_sample_stockfish_df()
        
        df_merged1 = merge_datasets(df_pgn, df_uci, df_sf)
        df_merged2 = merge_datasets(df_pgn, df_uci, df_sf)
        
        pd.testing.assert_frame_equal(df_merged1, df_merged2, check_dtype=True)


class TestFeatureValueRanges:
    """Test that extracted features are within valid ranges."""

    def test_acl_values_in_range(self, tmp_path):
        """Test that ACL (accuracy) values are between 0 and 100."""
        sf_file = tmp_path / "stockfish.csv"
        sf_content = """Event,MoveScores,EvalAtEnd
1,"[100, 50, -30]",0.5
2,"[50, -50, 0]",-0.3
"""
        sf_file.write_text(sf_content)
        
        df = extract_stockfish_features(str(sf_file))
        
        if "white_acl" in df.columns:
            assert (df["white_acl"] >= 0).all(), "ACL should be >= 0"
            assert (df["white_acl"] <= 100).all(), "ACL should be <= 100"

    def test_blunder_counts_non_negative(self, tmp_path):
        """Test that blunder counts are non-negative integers."""
        sf_file = tmp_path / "stockfish.csv"
        sf_content = """Event,MoveScores,EvalAtEnd
1,"[100, 50, -30]",0.5
"""
        sf_file.write_text(sf_content)
        
        df = extract_stockfish_features(str(sf_file))
        
        if "white_blunders" in df.columns:
            assert (df["white_blunders"] >= 0).all(), "Blunder count should be >= 0"


class TestTransformConsistency:
    """Test consistency across transformation operations."""

    def test_event_id_preserved_in_merge(self):
        """Test that event_id is preserved correctly through merge."""
        df_pgn = pd.DataFrame({
            "event_id": [1, 2, 3],
            "white_elo": [1600, 1400, 1800]
        })
        df_uci = pd.DataFrame({
            "event_id": [1, 2, 3],
            "moves_uci": ["e2e4", "d2d4", "g1f3"]
        })
        df_sf = pd.DataFrame({
            "event_id": [1, 2, 3],
            "white_acl": [85.5, 90.0, 88.3]
        })
        
        df_merged = merge_datasets(df_pgn, df_uci, df_sf)
        
        # Check event_ids match original
        assert df_merged["event_id"].equals(pd.Series([1, 2, 3])), "event_ids should be preserved"

    def test_elo_values_unchanged_in_merge(self):
        """Test that ELO values are not modified during merge."""
        df_pgn = pd.DataFrame({
            "event_id": [1, 2],
            "white_elo": [1600, 1400],
            "black_elo": [1500, 1600]
        })
        df_uci = pd.DataFrame({
            "event_id": [1, 2],
            "moves_uci": ["e2e4", "d2d4"]
        })
        df_sf = pd.DataFrame({
            "event_id": [1, 2],
            "white_acl": [85.5, 90.0]
        })
        
        df_merged = merge_datasets(df_pgn, df_uci, df_sf)
        
        # Check ELO values are preserved
        assert (df_merged["white_elo"] == [1600, 1400]).all(), "white_elo should be preserved"
        assert (df_merged["black_elo"] == [1500, 1600]).all(), "black_elo should be preserved"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
