import numpy as np
import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features and encode target variables.

    Parameters
    ----------
    df : merged DataFrame from merge_datasets()

    Returns
    -------
    DataFrame with additional engineered columns
    """
    df = df.copy()

    # ── Elo classification target (bin into skill tiers) ─────────────────────
    # white_elo is used ONLY to derive the target, then kept in dataset
    # but must be dropped before any model training (done in preprocessing notebook)
    elo_bins   = [0, 1000, 1500, 2000, 2500, 9999]
    elo_labels = ["Beginner", "Intermediate", "Advanced", "Expert", "Master"]

    df["elo_bucket_white"] = pd.cut(
        df["white_elo"], bins=elo_bins, labels=elo_labels, right=False
    )
    df["elo_bucket_black"] = pd.cut(
        df["black_elo"], bins=elo_bins, labels=elo_labels, right=False
    )

    # ── Ordinal encoding of target ────────────────────────────────────────────
    # Beginner=0, Intermediate=1, Advanced=2, Expert=3, Master=4
    ordinal_map = {"Beginner": 0, "Intermediate": 1, "Advanced": 2,
                   "Expert": 3, "Master": 4}
    df["elo_bucket_white_categorical"] = df["elo_bucket_white"].map(ordinal_map)
    df["elo_bucket_black_categorical"] = df["elo_bucket_black"].map(ordinal_map)

    # ── Accuracy gap ─────────────────────────────────────────────────────────
    df["acl_gap"] = (df["white_acl"] - df["black_acl"]).round(2)

    # ── Winner target variable ────────────────────────────────────────────────
    result_map_binary     = {"1-0": 1, "0-1": 0}
    result_map_multiclass = {"0-1": 0, "1/2-1/2": 1, "1-0": 2}

    df["winner_binary"]     = df["result"].map(result_map_binary)
    df["winner_multiclass"] = df["result"].map(result_map_multiclass)

    print("Engineered features added.")
    print(f"\nElo bucket distribution (White):\n{df['elo_bucket_white'].value_counts().sort_index().to_string()}")
    return df

def validation_report(df: pd.DataFrame) -> None:
    """
    Print a comprehensive data validation report.

    Covers:
        - Shape (rows, columns)
        - Data types per column
        - Missing values count and percentage
        - Duplicate rows
        - Class distribution for target variable
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

    print(f"\n── Duplicates ──────────────────────────────────────────────")
    dup_count = df.duplicated(subset=["event_id"]).sum() if "event_id" in df.columns else "N/A"
    print(f"Duplicate event_id rows: {dup_count}")

    print("\n── Numeric Summary ─────────────────────────────────────────")
    # white_elo included here for context only — not a model input feature
    numeric_cols = [c for c in ["white_elo", "black_elo", "num_moves", "white_acl",
                    "black_acl", "white_blunders", "black_blunders",
                    "acl_gap", "game_sharpness"] if c in df.columns]
    print(df[numeric_cols].describe().round(2).to_string())

    print("\n── Target: elo_bucket_white (categorical) ──────────────────")
    if "elo_bucket_white" in df.columns:
        print(df["elo_bucket_white"].value_counts().sort_index().to_string())

    print("\n── Target: elo_bucket_white_categorical (ordinal) ──────────────────")
    if "elo_bucket_white_categorical" in df.columns:
        label_map = {0:"Beginner",1:"Intermediate",2:"Advanced",3:"Expert",4:"Master"}
        print(df["elo_bucket_white_categorical"].value_counts().sort_index().rename(label_map).to_string())

    print("\n" + "=" * 60)