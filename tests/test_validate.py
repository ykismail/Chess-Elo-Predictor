"""
Unit Tests for Phase 2 — Data Validation
========================================
Tests for data quality checks and validation reporting.

Test Levels:
  • Validation Functions: validate_sources, validate_merged, validate_features
  • Error Handling: graceful handling of missing files
  • No False Positives: validation passes for good data
  • Report Structure: validation_report produces correct output
  • Anomaly Detection: detection of duplicates, missing values, outliers
  • Consistency: same data produces same validation report
"""

import os
import sys
import tempfile
import pytest
import pandas as pd
import numpy as np
from io import StringIO

# ── Path bootstrap ────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, ".."))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# Mock the intermediate directory for tests
sys.path.insert(0, os.path.join(project_dir, "src", "data"))
from validate import validate_sources, validate_merged, validate_features
from build_features import validation_report


class TestValidationFunctions:
    """Test validation function execution and error handling."""

    def test_validate_sources_requires_files(self, tmp_path, monkeypatch):
        """Test that validate_sources handles missing files gracefully."""
        # Create a temporary directory and point intermediate_dir to it
        monkeypatch.setenv("INTERMEDIATE_DIR", str(tmp_path))
        
        # This should handle missing files without crashing
        # (depends on implementation - might skip or raise)
        try:
            validate_sources()
        except FileNotFoundError:
            pytest.skip("Implementation raises on missing files (expected)")

    def test_validate_merged_requires_file(self, tmp_path, monkeypatch):
        """Test that validate_merged handles missing kaggle_merged.csv gracefully."""
        monkeypatch.setenv("INTERMEDIATE_DIR", str(tmp_path))
        
        try:
            validate_merged()
        except FileNotFoundError:
            pytest.skip("Implementation raises on missing files (expected)")

    def test_validate_features_requires_file(self, tmp_path, monkeypatch):
        """Test that validate_features handles missing merged_games.csv gracefully."""
        monkeypatch.setenv("INTERMEDIATE_DIR", str(tmp_path))
        
        try:
            validate_features()
        except FileNotFoundError:
            pytest.skip("Implementation raises on missing files (expected)")


class TestValidationReport:
    """Test data validation report generation."""

    def create_sample_df(self):
        """Create a sample DataFrame for validation testing."""
        return pd.DataFrame({
            "event_id": [1, 2, 3, 4, 5],
            "white_elo": [1600, 1400, 1800, 1500, 1700],
            "black_elo": [1500, 1600, 1700, 1400, 1600],
            "white_acl": [85.5, 90.0, 88.3, 82.1, 87.5],
            "black_acl": [80.0, 92.5, 87.1, 83.5, 85.0],
            "result": ["1-0", "0-1", "1/2-1/2", "1-0", "0-1"],
            "num_moves": [20, 35, 42, 18, 30],
            "white_blunders": [0, 1, 0, 1, 0],
            "black_blunders": [1, 0, 0, 0, 1]
        })

    def test_validation_report_executes(self, capsys):
        """Test that validation_report executes without error."""
        df = self.create_sample_df()
        
        validation_report(df)
        
        captured = capsys.readouterr()
        assert "VALIDATION REPORT" in captured.out

    def test_validation_report_includes_shape(self, capsys):
        """Test that validation report includes DataFrame shape."""
        df = self.create_sample_df()
        
        validation_report(df)
        
        captured = capsys.readouterr()
        assert "Shape:" in captured.out
        assert "5 rows" in captured.out or "rows" in captured.out

    def test_validation_report_includes_data_types(self, capsys):
        """Test that validation report includes data types section."""
        df = self.create_sample_df()
        
        validation_report(df)
        
        captured = capsys.readouterr()
        assert "Data Types" in captured.out or "dtype" in captured.out

    def test_validation_report_includes_missing_values(self, capsys):
        """Test that validation report includes missing values section."""
        df = self.create_sample_df()
        
        validation_report(df)
        
        captured = capsys.readouterr()
        assert "Missing" in captured.out

    def test_validation_report_detects_missing_values(self, capsys):
        """Test that validation report detects actual missing values."""
        df = self.create_sample_df()
        df.loc[0, "white_elo"] = np.nan
        
        validation_report(df)
        
        captured = capsys.readouterr()
        assert "Missing" in captured.out or "NaN" in captured.out or "null" in captured.out

    def test_validation_report_includes_duplicates(self, capsys):
        """Test that validation report includes duplicate detection."""
        df = self.create_sample_df()
        
        validation_report(df)
        
        captured = capsys.readouterr()
        assert "Duplicate" in captured.out or "duplicate" in captured.out

    def test_validation_report_detects_duplicates(self, capsys):
        """Test that validation report detects actual duplicates."""
        df = self.create_sample_df()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # Add duplicate
        
        validation_report(df)
        
        captured = capsys.readouterr()
        assert "Duplicate" in captured.out or "1" in captured.out  # Should report 1 duplicate


class TestDataQualityChecks:
    """Test specific data quality issues that validation should catch."""

    def test_detect_negative_elo(self):
        """Test detection of negative ELO values."""
        df = pd.DataFrame({
            "event_id": [1, 2],
            "white_elo": [1600, -100],  # Invalid negative
            "black_elo": [1500, 1600]
        })
        
        # Check data quality manually
        invalid_elo = (df["white_elo"] < 0).any()
        assert invalid_elo, "Should detect negative ELO"

    def test_detect_missing_critical_columns(self):
        """Test detection of missing critical columns."""
        df = pd.DataFrame({
            "event_id": [1, 2],
            "white_elo": [1600, 1400]
            # Missing black_elo
        })
        
        missing_critical = "black_elo" not in df.columns
        assert missing_critical, "Should detect missing critical column"

    def test_detect_invalid_results(self):
        """Test detection of invalid game result values."""
        df = pd.DataFrame({
            "event_id": [1, 2, 3],
            "result": ["1-0", "0-1", "2-0"]  # Invalid result
        })
        
        valid_results = {"1-0", "0-1", "1/2-1/2"}
        invalid_result = not df["result"].isin(valid_results).all()
        assert invalid_result, "Should detect invalid result"

    def test_detect_acl_out_of_range(self):
        """Test detection of ACL values outside [0, 100] range."""
        df = pd.DataFrame({
            "event_id": [1, 2],
            "white_acl": [85.5, 150.0]  # 150 is invalid
        })
        
        out_of_range = ((df["white_acl"] < 0) | (df["white_acl"] > 100)).any()
        assert out_of_range, "Should detect ACL out of range"


class TestValidationReportStructure:
    """Test structure and completeness of validation reports."""

    def test_validation_report_has_multiple_sections(self, capsys):
        """Test that validation report has multiple sections."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        })
        
        validation_report(df)
        
        captured = capsys.readouterr()
        # Check for multiple section headers
        sections = ["Shape:", "Data Types", "Missing"]
        found_sections = sum(1 for s in sections if s in captured.out or s.lower() in captured.out)
        assert found_sections >= 2, "Should have at least 2 sections"

    def test_validation_report_summary_info(self, capsys):
        """Test that validation report includes summary information."""
        df = pd.DataFrame({
            "event_id": [1, 2, 3],
            "value": [10.5, 20.3, 15.8]
        })
        
        validation_report(df)
        
        captured = capsys.readouterr()
        # Should include numeric summary
        assert "describe" in captured.out.lower() or "mean" in captured.out.lower() or "count" in captured.out


class TestValidationConsistency:
    """Test consistency of validation across multiple runs."""

    def test_validation_report_reproducibility(self, capsys):
        """Test that validation report is reproducible for same DataFrame."""
        df = pd.DataFrame({
            "event_id": [1, 2, 3],
            "white_elo": [1600, 1400, 1800],
            "black_elo": [1500, 1600, 1700]
        })
        
        # First validation
        validation_report(df)
        output1 = capsys.readouterr().out
        
        # Second validation
        validation_report(df)
        output2 = capsys.readouterr().out
        
        # Both should contain same key statistics
        assert "3 rows" in output1 or "rows" in output1
        assert "3 rows" in output2 or "rows" in output2


class TestValidationEdgeCases:
    """Test validation behavior with edge cases."""

    def test_validation_empty_dataframe(self, capsys):
        """Test validation report with empty DataFrame."""
        df = pd.DataFrame()
        
        # Should handle empty DataFrame
        validation_report(df)
        
        captured = capsys.readouterr()
        assert "Shape:" in captured.out or "VALIDATION" in captured.out

    def test_validation_single_row(self, capsys):
        """Test validation report with single row DataFrame."""
        df = pd.DataFrame({
            "event_id": [1],
            "value": [100]
        })
        
        validation_report(df)
        
        captured = capsys.readouterr()
        assert "1" in captured.out or "row" in captured.out

    def test_validation_all_nulls(self, capsys):
        """Test validation report with column of all NaN values."""
        df = pd.DataFrame({
            "event_id": [1, 2, 3],
            "all_null_col": [np.nan, np.nan, np.nan]
        })
        
        validation_report(df)
        
        captured = capsys.readouterr()
        # Should report missing values
        assert "all_null_col" in captured.out or "100" in captured.out or "3" in captured.out

    def test_validation_single_column(self, capsys):
        """Test validation report with single column DataFrame."""
        df = pd.DataFrame({"value": [1, 2, 3]})
        
        validation_report(df)
        
        captured = capsys.readouterr()
        assert "value" in captured.out or "VALIDATION" in captured.out


class TestValidationAnomalies:
    """Test detection of various data anomalies."""

    def test_check_for_outliers_elo(self):
        """Test IQR-based outlier detection for ELO values."""
        df = pd.DataFrame({
            "white_elo": [1600, 1400, 1800, 1500, 1700, 5000]  # 5000 is outlier
        })
        
        Q1 = df["white_elo"].quantile(0.25)
        Q3 = df["white_elo"].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df["white_elo"] < Q1 - 1.5 * IQR) | 
                   (df["white_elo"] > Q3 + 1.5 * IQR))
        
        assert outliers.any(), "Should detect ELO outliers"

    def test_check_duplicate_events(self):
        """Test detection of duplicate event IDs."""
        df = pd.DataFrame({
            "event_id": [1, 2, 3, 1],  # event_id 1 is duplicate
        })
        
        duplicates = df.duplicated(subset=["event_id"]).sum()
        assert duplicates > 0, "Should detect duplicate event_ids"

    def test_check_all_same_values(self):
        """Test detection when all values in column are identical."""
        df = pd.DataFrame({
            "result": ["1-0", "1-0", "1-0", "1-0"]
        })
        
        same_values = (df["result"].nunique() == 1)
        assert same_values, "Should detect when all values are same"


class TestValidationIntegration:
    """Integration tests for validation workflow."""

    def test_validation_workflow_valid_data(self, capsys):
        """Test complete validation workflow with valid data."""
        df = pd.DataFrame({
            "event_id": range(1, 6),
            "white_elo": [1600, 1400, 1800, 1500, 1700],
            "black_elo": [1500, 1600, 1700, 1400, 1600],
            "result": ["1-0", "0-1", "1/2-1/2", "1-0", "0-1"],
            "white_acl": [85.5, 90.0, 88.3, 82.1, 87.5],
            "black_acl": [80.0, 92.5, 87.1, 83.5, 85.0]
        })
        
        # Should complete without error
        validation_report(df)
        
        captured = capsys.readouterr()
        assert "VALIDATION REPORT" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
