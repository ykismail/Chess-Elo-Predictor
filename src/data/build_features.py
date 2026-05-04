"""
Phase 4 — Feature Engineering
==============================
Responsibilities:
  1. engineer_features()   — derive Elo buckets, ordinal encoding, acl_gap,
                              winner targets on the Kaggle merged dataset.
  2. integrate_datasets()  — concatenate engineered Kaggle data with the
                              harmonised Lichess dataset (load_lichess runs
                              its own feature engineering internally).
  3. Save final outputs:
       data/intermediate/games.csv         — Kaggle-only engineered dataset
       data/intermediate/merged_games.csv  — full combined dataset (final output)

All feature extraction that was scattered in data_loading.py
(engineer_features call, load_lichess call, pd.concat, event_id reassignment)
lives here now, as originally noted by the TODO comments #3 and #4.
"""

import os
import sys

# ── Path bootstrap ────────────────────────────────────────────────────────────
script_dir  = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

import numpy as np
import pandas as pd
from load_data import integrate_datasets

# ── Directory paths ───────────────────────────────────────────────────────────
raw_dir          = os.path.join(project_dir, "data", "raw")
intermediate_dir = os.path.join(project_dir, "data", "intermediate")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Feature engineering on Kaggle data
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features and encode target variables on the merged Kaggle data.

    Applied to the output of Phase 3 (kaggle_merged.csv).

    New columns
    -----------
    elo_bucket_white / elo_bucket_black         : categorical skill tier
    elo_bucket_white_categorical / _black_*     : ordinal int (0–4)
    acl_gap                                     : white_acl − black_acl
    winner_binary                               : 1=White wins, 0=Black wins
    winner_multiclass                           : 0=Black, 1=Draw, 2=White

    Note: white_elo / black_elo are kept for reference but must be dropped
    before model training (done in the preprocessing notebook).
    """
    df = df.copy()

    # ── Elo skill-tier classification target ──────────────────────────────────
    elo_bins   = [0, 1000, 1500, 2000, 2500, 9999]
    elo_labels = ["Beginner", "Intermediate", "Advanced", "Expert", "Master"]

    df["elo_bucket_white"] = pd.cut(
        df["white_elo"], bins=elo_bins, labels=elo_labels, right=False
    )
    df["elo_bucket_black"] = pd.cut(
        df["black_elo"], bins=elo_bins, labels=elo_labels, right=False
    )

    # ── Ordinal encoding of skill tier ───────────────────────────────────────
    ordinal_map = {
        "Beginner": 0, "Intermediate": 1, "Advanced": 2,
        "Expert": 3, "Master": 4,
    }
    df["elo_bucket_white_categorical"] = df["elo_bucket_white"].map(ordinal_map)
    df["elo_bucket_black_categorical"] = df["elo_bucket_black"].map(ordinal_map)

    # ── Accuracy gap ──────────────────────────────────────────────────────────
    df["acl_gap"] = (df["white_acl"] - df["black_acl"]).round(2)

    # ── Winner target variables ───────────────────────────────────────────────
    df["winner_binary"]     = df["result"].map({"1-0": 1, "0-1": 0})
    df["winner_multiclass"] = df["result"].map({"0-1": 0, "1/2-1/2": 1, "1-0": 2})

    print("Engineered features added to Kaggle data.")
    print(
        f"\nElo bucket distribution (White):\n"
        f"{df['elo_bucket_white'].value_counts().sort_index().to_string()}"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Validation report  (kept here so build_features is self-contained)
# ─────────────────────────────────────────────────────────────────────────────

def validation_report(df: pd.DataFrame) -> None:
    """
    Print a comprehensive data validation report covering shape, dtypes,
    missing values, duplicates, numeric summary, outliers (IQR), and
    target-class distributions.
    """
    print("=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)

    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    print("\n── Data Types ──────────────────────────────────────────────")
    print(df.dtypes.to_string())

    print("\n── Missing Values ──────────────────────────────────────────")
    missing     = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df  = pd.DataFrame({"count": missing, "pct": missing_pct})
    missing_df  = missing_df[missing_df["count"] > 0]
    if missing_df.empty:
        print("No missing values found.")
    else:
        print(missing_df.to_string())

    print("\n── Duplicates ──────────────────────────────────────────────")
    dup_count = (
        df.duplicated(subset=["event_id"]).sum()
        if "event_id" in df.columns
        else "N/A"
    )
    print(f"Duplicate event_id rows: {dup_count}")

    print("\n── Numeric Summary ─────────────────────────────────────────")
    numeric_cols = [
        c for c in [
            "white_elo", "black_elo", "num_moves", "white_acl",
            "black_acl", "white_blunders", "black_blunders",
            "acl_gap", "game_sharpness",
        ]
        if c in df.columns
    ]
    if numeric_cols:
        print(df[numeric_cols].describe().round(2).to_string())

        print("\n── Outliers (IQR Method) ───────────────────────────────────")
        outliers = {}
        for col in numeric_cols:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR    = Q3 - Q1
            mask   = (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)
            outliers[col] = mask.sum()
        out_df = pd.DataFrame(
            list(outliers.items()), columns=["Column", "Outliers count"]
        )
        out_df["Pct (%)"] = (out_df["Outliers count"] / len(df) * 100).round(2)
        out_df = out_df[out_df["Outliers count"] > 0]
        print(
            "No outliers detected." if out_df.empty
            else out_df.to_string(index=False)
        )
    else:
        print("No numeric columns found.")

    print("\n── Target: elo_bucket_white (categorical) ──────────────────")
    if "elo_bucket_white" in df.columns:
        print(df["elo_bucket_white"].value_counts().sort_index().to_string())

    print("\n── Target: elo_bucket_white_categorical (ordinal) ──────────")
    if "elo_bucket_white_categorical" in df.columns:
        label_map = {0: "Beginner", 1: "Intermediate", 2: "Advanced",
                     3: "Expert",   4: "Master"}
        print(
            df["elo_bucket_white_categorical"]
            .value_counts()
            .sort_index()
            .rename(label_map)
            .to_string()
        )

    print("\n" + "=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Build full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_pipeline() -> pd.DataFrame:
    """
    Full Phase-4 pipeline:

      kaggle_merged.csv
          → engineer_features()          adds Elo buckets, acl_gap, winner cols
          → source / has_stockfish tags  (was inline in data_loading.py)
          → integrate_datasets()         appends Lichess data (load_lichess
                                         runs its own feature engineering)
          → reassign event_id            sequential 1…N across combined dataset

    Returns
    -------
    df_combined : fully engineered combined DataFrame
    """
    # ── Load Phase-3 output ───────────────────────────────────────────────────
    kaggle_merged_path = os.path.join(intermediate_dir, "kaggle_merged.csv")
    print(f"Loading {kaggle_merged_path}...")
    df_kaggle = pd.read_csv(kaggle_merged_path)
    print(f"  Loaded shape : {df_kaggle.shape}")

    # ── Feature engineering on Kaggle partition ───────────────────────────────
    # (was commented block #3 in data_loading.py)
    print("\nRunning engineer_features() on Kaggle data...")
    df_kaggle_engineered = engineer_features(df_kaggle)

    # Tag the Kaggle partition — these two lines were in data_loading.py
    # between engineer_features and load_lichess
    df_kaggle_engineered["source"]        = "kaggle_pgn"
    df_kaggle_engineered["has_stockfish"] = True

    # ── Save Kaggle-only subset ───────────────────────────────────────────────
    games_path = os.path.join(intermediate_dir, "games.csv")
    df_kaggle_engineered.to_csv(games_path, index=False)
    print(f"\nSaved Kaggle-only dataset → {games_path}  {df_kaggle_engineered.shape}")

    # ── Integrate Lichess data ────────────────────────────────────────────────
    # (was commented block #4 in data_loading.py)
    # integrate_datasets() calls load_lichess() internally, which performs
    # its own harmonisation + feature engineering for the Lichess partition.
    chess_games_path  = os.path.join(raw_dir, "chess_games.csv")
    lichess_sf_path   = os.path.join(raw_dir, "lichess_stockfish.csv")

    print("\nIntegrating Lichess dataset...")
    df_combined = integrate_datasets(
        df_pgn_pipeline=df_kaggle_engineered,
        lichess_path=chess_games_path,
        lichess_sf_path=lichess_sf_path,
    )
    # event_id is already reassigned sequentially inside integrate_datasets()

    # ── Save final merged dataset ─────────────────────────────────────────────
    merged_path = os.path.join(intermediate_dir, "merged_games.csv")
    df_combined.to_csv(merged_path, index=False)
    print(f"\nSaved final merged dataset → {merged_path}  {df_combined.shape}")

    return df_combined


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df_combined = build_feature_pipeline()

    print("\n── Final Dataset Validation ─────────────────────────────────")
    validation_report(df_combined)

    print(f"\n✓ Phase 4 Complete.")
    print(f"  Final dataset shape : {df_combined.shape}")
    print(f"  Output              : data/intermediate/merged_games.csv")