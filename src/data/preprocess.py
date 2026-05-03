"""
src/data/preprocess.py
Preprocessing pipeline for the chess skill classification project.

Functions
---------
load_and_split   — load merged_games.csv, drop leaking cols, split train/val/test
scale_features   — fit StandardScaler on train, transform val and test
encode_cats      — encode categorical columns
impute_missing   — median imputation for NaN Stockfish values
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

# Columns derived from white_elo — must be dropped to prevent data leakage
LEAKAGE_COLS = [
    "white_elo", "elo_gap", "avg_elo",
    "elo_bucket_white", "elo_bucket_black",
    "elo_bucket_black_enc",
    "winner_binary", "winner_multiclass",
    "event_id", "source",
    "moves_san", "moves_uci", "moves_pgn",
    "opening_name",
]

TARGET_COL   = "elo_bucket_white_enc"
TARGET_NAMES = ["Beginner", "Intermediate", "Advanced", "Expert", "Master"]


def encode_cats(X: pd.DataFrame) -> pd.DataFrame:
    """
    Encode all categorical columns in the feature matrix.

    Encoding scheme
    ---------------
    termination        : OrdinalEncoder with fixed category order
    eco_family         : LabelEncoder (A-E + unknown)
    eco_code           : LabelEncoder (hundreds of unique codes)
    white/black_castle_side : OrdinalEncoder (none < kingside < queenside)
    white/black_castled     : bool → int
    has_stockfish           : bool → int

    Parameters
    ----------
    X : pd.DataFrame — raw feature matrix (leakage cols already dropped)

    Returns
    -------
    pd.DataFrame with all categoricals encoded as integers
    """
    X = X.copy()

    # Boolean columns → int
    for col in ["white_castled", "black_castled", "has_stockfish"]:
        if col in X.columns:
            X[col] = X[col].astype("boolean").astype("Int64")

    # Termination
    if "termination" in X.columns:
        oe = OrdinalEncoder(
            categories=[["checkmate", "resignation", "draw", "timeout", "unknown"]],
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        X["termination"] = oe.fit_transform(X[["termination"]])

    # ECO family
    if "eco_family" in X.columns:
        X["eco_family"] = X["eco_family"].fillna("?")
        le = LabelEncoder()
        X["eco_family"] = le.fit_transform(X["eco_family"].astype(str))

    # ECO code
    if "eco_code" in X.columns:
        X["eco_code"] = X["eco_code"].fillna("Unknown")
        le = LabelEncoder()
        X["eco_code"] = le.fit_transform(X["eco_code"].astype(str))

    # Castle sides
    for col in ["white_castle_side", "black_castle_side"]:
        if col in X.columns:
            X[col] = X[col].fillna("none")
            oe = OrdinalEncoder(
                categories=[["none", "kingside", "queenside"]],
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            X[col] = oe.fit_transform(X[[col]])

    return X


def impute_missing(X_train, X_val, X_test):
    """
    Median imputation for missing Stockfish values (18 Lichess games).
    Fit on training set only — applied to val and test.

    Returns
    -------
    X_train, X_val, X_test with NaNs filled
    """
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    imputer  = SimpleImputer(strategy="median")

    X_train = X_train.copy()
    X_val   = X_val.copy()
    X_test  = X_test.copy()

    X_train[num_cols] = imputer.fit_transform(X_train[num_cols])
    X_val[num_cols]   = imputer.transform(X_val[num_cols])
    X_test[num_cols]  = imputer.transform(X_test[num_cols])

    return X_train, X_val, X_test


def scale_features(X_train, X_val, X_test):
    """
    StandardScaler — fit on train, transform val and test.

    Returns
    -------
    X_train_scaled, X_val_scaled, X_test_scaled, fitted scaler
    """
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    scaler   = StandardScaler()

    X_train_s = X_train.copy()
    X_val_s   = X_val.copy()
    X_test_s  = X_test.copy()

    X_train_s[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val_s[num_cols]   = scaler.transform(X_val[num_cols])
    X_test_s[num_cols]  = scaler.transform(X_test[num_cols])

    return X_train_s, X_val_s, X_test_s, scaler


def load_and_split(filepath: str, random_state: int = 42):
    """
    Full preprocessing pipeline:
        1. Load merged_games.csv
        2. Drop leaking and irrelevant columns
        3. Encode categoricals
        4. Impute missing values
        5. Stratified 70/15/15 train/val/test split
        6. Scale numeric features

    Parameters
    ----------
    filepath     : str — path to merged_games.csv
    random_state : int — random seed for reproducibility

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test, scaler
    """
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns from '{filepath}'")

    # Separate target
    y = df[TARGET_COL].copy()

    # Drop leakage and non-feature columns
    cols_to_drop = [c for c in LEAKAGE_COLS + [TARGET_COL] if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    print(f"Features after dropping leakage cols: {X.shape[1]}")

    # Encode
    X = encode_cats(X)

    # Split (before scaling to prevent leakage)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )

    # Impute
    X_train, X_val, X_test = impute_missing(X_train, X_val, X_test)

    # Scale
    X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test)

    print(f"\nSplit: Train {len(X_train):,} | Val {len(X_val):,} | Test {len(X_test):,}")
    print(f"Target distribution (train):")
    counts = y_train.value_counts().sort_index()
    for enc, label in enumerate(TARGET_NAMES):
        n = counts.get(enc, 0)
        print(f"  {label:<15}: {n:>6,}  ({n/len(y_train)*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler
