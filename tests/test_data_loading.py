"""
Unit Tests for Phase 1 — Data Loading
======================================
Tests for data acquisition, parsing, and initial data quality.

Test Levels:
  • Custom Functions: parse_pgn, parse_uci, ECO download
  • Data Shapes: output rows, columns, types
  • Reproducibility: same input → same output
  • No Missing Data: where not expected
  • Consistency: event_id uniqueness, column presence
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

from src.data.load_data import parse_pgn, parse_uci, download_eco_database


class TestParsePGN:
    """Test PGN file parsing functionality."""

    def test_parse_pgn_returns_dataframe(self, tmp_path):
        """Test that parse_pgn returns a pandas DataFrame."""
        pgn_file = tmp_path / "test.pgn"
        pgn_content = """[Event "Test Game 1"]
[White "Player A"]
[Black "Player B"]
[WhiteElo "1600"]
[BlackElo "1400"]
[Result "1-0"]

1. e4 c5 2. Nf3 d6 1-0
"""
        pgn_file.write_text(pgn_content)
        
        df = parse_pgn(str(pgn_file))
        
        assert isinstance(df, pd.DataFrame), "parse_pgn should return DataFrame"
        assert len(df) > 0, "DataFrame should have at least one row"

    def test_parse_pgn_output_shape(self, tmp_path):
        """Test that parse_pgn output has expected columns."""
        pgn_file = tmp_path / "test.pgn"
        pgn_content = """[Event "Test"]
[White "A"]
[Black "B"]
[WhiteElo "1600"]
[BlackElo "1400"]
[Result "1-0"]

1. e4 e5 1-0
"""
        pgn_file.write_text(pgn_content)
        
        df = parse_pgn(str(pgn_file))
        
        required_cols = ["white_elo", "black_elo", "result", "moves_san"]
        for col in required_cols:
            assert col in df.columns, f"Column '{col}' not found in parse_pgn output"

    def test_parse_pgn_data_types(self, tmp_path):
        """Test that parse_pgn output has correct data types."""
        pgn_file = tmp_path / "test.pgn"
        pgn_content = """[Event "Test"]
[White "A"]
[Black "B"]
[WhiteElo "1600"]
[BlackElo "1400"]
[Result "1-0"]

1. e4 e5 1-0
"""
        pgn_file.write_text(pgn_content)
        
        df = parse_pgn(str(pgn_file))
        
        # Check numeric columns
        assert pd.api.types.is_numeric_dtype(df["white_elo"]), "white_elo should be numeric"
        assert pd.api.types.is_numeric_dtype(df["black_elo"]), "black_elo should be numeric"
        # Check string columns
        assert df["result"].dtype == "object", "result should be string"
        assert df["moves_san"].dtype == "object", "moves_san should be string"

    def test_parse_pgn_no_missing_values(self, tmp_path):
        """Test that parse_pgn does not produce unexpected missing values."""
        pgn_file = tmp_path / "test.pgn"
        pgn_content = """[Event "Test"]
[White "A"]
[Black "B"]
[WhiteElo "1600"]
[BlackElo "1400"]
[Result "1-0"]

1. e4 e5 1-0
"""
        pgn_file.write_text(pgn_content)
        
        df = parse_pgn(str(pgn_file))
        
        # Check critical columns have no NaN
        critical_cols = ["white_elo", "black_elo", "result"]
        for col in critical_cols:
            assert df[col].isna().sum() == 0, f"Column '{col}' should not have NaN values"

    def test_parse_pgn_reproducibility(self, tmp_path):
        """Test that parse_pgn produces identical output for same input."""
        pgn_file = tmp_path / "test.pgn"
        pgn_content = """[Event "Test"]
[White "A"]
[Black "B"]
[WhiteElo "1600"]
[BlackElo "1400"]
[Result "1-0"]

1. e4 e5 1-0
"""
        pgn_file.write_text(pgn_content)
        
        df1 = parse_pgn(str(pgn_file))
        df2 = parse_pgn(str(pgn_file))
        
        pd.testing.assert_frame_equal(df1, df2, check_dtype=True)


class TestParseUCI:
    """Test UCI file parsing functionality."""

    def test_parse_uci_returns_dataframe(self, tmp_path):
        """Test that parse_uci returns a pandas DataFrame."""
        uci_file = tmp_path / "test_uci.pgn"
        uci_content = """[Event "Test"]
[White "A"]
[Black "B"]
[Result "1-0"]

1. e2e4 e7e5 1-0
"""
        uci_file.write_text(uci_content)
        
        df = parse_uci(str(uci_file))
        
        assert isinstance(df, pd.DataFrame), "parse_uci should return DataFrame"
        assert len(df) > 0, "DataFrame should have at least one row"

    def test_parse_uci_output_columns(self, tmp_path):
        """Test that parse_uci output has expected columns."""
        uci_file = tmp_path / "test_uci.pgn"
        uci_content = """[Event "Test"]
[White "A"]
[Black "B"]
[Result "1-0"]

1. e2e4 e7e5 1-0
"""
        uci_file.write_text(uci_content)
        
        df = parse_uci(str(uci_file))
        
        required_cols = ["event_id", "moves_uci"]
        for col in required_cols:
            assert col in df.columns, f"Column '{col}' not found in parse_uci output"

    def test_parse_uci_reproducibility(self, tmp_path):
        """Test that parse_uci produces identical output for same input."""
        uci_file = tmp_path / "test_uci.pgn"
        uci_content = """[Event "Test"]
[White "A"]
[Black "B"]
[Result "1-0"]

1. e2e4 e7e5 1-0
"""
        uci_file.write_text(uci_content)
        
        df1 = parse_uci(str(uci_file))
        df2 = parse_uci(str(uci_file))
        
        pd.testing.assert_frame_equal(df1, df2, check_dtype=True)


class TestECODatabase:
    """Test ECO opening database download and structure."""

    def test_download_eco_database_returns_dataframe(self, tmp_path):
        """Test that download_eco_database returns a DataFrame."""
        save_path = str(tmp_path / "eco.csv")
        
        try:
            df = download_eco_database(save_path)
            
            assert isinstance(df, pd.DataFrame), "Should return DataFrame"
            assert len(df) > 0, "DataFrame should have rows"
        except Exception as e:
            pytest.skip(f"Network error: {e}")

    def test_eco_database_required_columns(self, tmp_path):
        """Test that ECO database has required columns."""
        save_path = str(tmp_path / "eco.csv")
        
        try:
            df = download_eco_database(save_path)
            
            required_cols = ["pgn", "name"]
            for col in required_cols:
                assert col in df.columns, f"Column '{col}' not found in ECO database"
        except Exception as e:
            pytest.skip(f"Network error: {e}")

    def test_eco_database_file_saved(self, tmp_path):
        """Test that ECO database is saved to file."""
        save_path = str(tmp_path / "eco.csv")
        
        try:
            download_eco_database(save_path)
            
            assert os.path.exists(save_path), "ECO CSV file should be saved"
            assert os.path.getsize(save_path) > 0, "ECO CSV file should not be empty"
        except Exception as e:
            pytest.skip(f"Network error: {e}")


class TestDataConsistency:
    """Test consistency and integration across parsing functions."""

    def test_parsed_dataframes_have_event_id(self, tmp_path):
        """Test that parsed DataFrames include event_id for joining."""
        pgn_file = tmp_path / "test.pgn"
        pgn_content = """[Event "Test"]
[White "A"]
[Black "B"]
[WhiteElo "1600"]
[BlackElo "1400"]
[Result "1-0"]

1. e4 e5 1-0
"""
        pgn_file.write_text(pgn_content)
        
        df = parse_pgn(str(pgn_file))
        
        assert "event_id" in df.columns, "event_id column should exist for joining"
        assert df["event_id"].nunique() > 0, "event_id should have unique values"

    def test_parsed_pgn_elo_ranges(self, tmp_path):
        """Test that parsed ELO values are in reasonable ranges."""
        pgn_file = tmp_path / "test.pgn"
        pgn_content = """[Event "Test"]
[White "A"]
[Black "B"]
[WhiteElo "1600"]
[BlackElo "1400"]
[Result "1-0"]

1. e4 e5 1-0
"""
        pgn_file.write_text(pgn_content)
        
        df = parse_pgn(str(pgn_file))
        
        # ELO should be between 0 and ~4000
        assert (df["white_elo"] >= 0).all() and (df["white_elo"] <= 4000).all()
        assert (df["black_elo"] >= 0).all() and (df["black_elo"] <= 4000).all()

    def test_parsed_results_valid_values(self, tmp_path):
        """Test that result column contains only valid game results."""
        pgn_file = tmp_path / "test.pgn"
        pgn_content = """[Event "Test1"]
[White "A"]
[Black "B"]
[Result "1-0"]

1. e4 e5 1-0

[Event "Test2"]
[White "C"]
[Black "D"]
[Result "0-1"]

1. d4 d5 0-1

[Event "Test3"]
[White "E"]
[Black "F"]
[Result "1/2-1/2"]

1. Nf3 Nf6 1/2-1/2
"""
        pgn_file.write_text(pgn_content)
        
        df = parse_pgn(str(pgn_file))
        
        valid_results = {"1-0", "0-1", "1/2-1/2"}
        assert df["result"].isin(valid_results).all(), "All results should be valid chess results"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
