"""
Phase 2 — Data Validation
=========================
Responsibilities:
  Run quality checks on every intermediate dataset produced by each
  pipeline phase and print a structured report.

  Call validate_sources()  after Phase 1 (data_loading.py)
  Call validate_merged()   after Phase 3 (transform_data.py)
  Call validate_features() after Phase 4 (build_features.py)

No data is written to disk — this phase is read-only.
"""

import os
import sys

# ── Path bootstrap ────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

import pandas as pd
from src.features.build_features import validation_report

# ── Directory paths ───────────────────────────────────────────────────────────
intermediate_dir = os.path.join(project_dir, "data", "intermediate")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  VALIDATING: {title}")
    print(f"{'=' * 60}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase-specific validation entry points
# ─────────────────────────────────────────────────────────────────────────────


def validate_sources() -> None:
    """
    Validate the two raw parsed DataFrames produced by Phase 1.

    Reads
    -----
    intermediate/parsed_data_uci.csv
    intermediate/parsed_data_pgn.csv
    """
    parsed_data_uci = pd.read_csv(os.path.join(intermediate_dir, "parsed_data_uci.csv"))
    parsed_data_pgn = pd.read_csv(os.path.join(intermediate_dir, "parsed_data_pgn.csv"))

    _header("parsed_data_uci  (Phase 1 output)")
    validation_report(parsed_data_uci)

    _header("parsed_data_pgn  (Phase 1 output)")
    validation_report(parsed_data_pgn)


def validate_merged() -> None:
    """
    Validate the Kaggle-merged dataset produced by Phase 3.

    Reads
    -----
    intermediate/kaggle_merged.csv
    """
    df_kaggle_merged = pd.read_csv(os.path.join(intermediate_dir, "kaggle_merged.csv"))

    _header("kaggle_merged  (Phase 3 output)")
    validation_report(df_kaggle_merged)

    # ── Source-level join audit ───────────────────────────────────────────────
    print("\n── Join Audit ───────────────────────────────────────────────")
    print(f"Total rows after merge  : {len(df_kaggle_merged):,}")
    if "event_id" in df_kaggle_merged.columns:
        n_unique = df_kaggle_merged["event_id"].nunique()
        print(f"Unique event_ids        : {n_unique:,}")
        n_dup = df_kaggle_merged.duplicated(subset=["event_id"]).sum()
        print(f"Duplicate event_ids     : {n_dup:,}")


def validate_features() -> None:
    """
    Validate the fully engineered dataset produced by Phase 4.

    Reads
    -----
    intermediate/merged_games.csv
    """
    df_merged_games = pd.read_csv(os.path.join(intermediate_dir, "merged_games.csv"))

    _header("merged_games  (Phase 4 output — final engineered dataset)")
    validation_report(df_merged_games)

    # ── Source breakdown ──────────────────────────────────────────────────────
    if "source" in df_merged_games.columns:
        print("\n── Source breakdown ─────────────────────────────────────────")
        print(df_merged_games["source"].value_counts().to_string())

    # ── Stockfish coverage ────────────────────────────────────────────────────
    if "has_stockfish" in df_merged_games.columns:
        print("\n── Stockfish coverage ───────────────────────────────────────")
        print(df_merged_games["has_stockfish"].value_counts().to_string())


# ─────────────────────────────────────────────────────────────────────────────
# Entry point  — runs all three checkpoints in sequence
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run data validation checkpoints.")
    parser.add_argument(
        "--phase",
        choices=["sources", "merged", "features", "all"],
        default="all",
        help="Which checkpoint to run (default: all).",
    )
    args = parser.parse_args()

    if args.phase in ("sources", "all"):
        validate_sources()

    if args.phase in ("merged", "all"):
        validate_merged()

    if args.phase in ("features", "all"):
        validate_features()

    print("\n✓ Phase 2 — Validation Complete.")
