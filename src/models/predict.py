"""
predict.py
==========
Loads the best trained model and runs predictions on the test set
or on new input data.

Usage
-----
    # Predict on the held-out test set (default)
    python src/models/predict.py

    # Predict on a custom CSV file
    python src/models/predict.py --input path/to/new_games.csv

    # Use a specific model (default: random_forest_track_b)
    python src/models/predict.py --model xgboost_track_b
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "merged_games.csv"
CONFIG_PATH = ROOT / "configs" / "model_params.json"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports" / "results"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_ORDER = ["Beginner", "Intermediate", "Advanced", "Expert", "Master"]


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def load_model(model_name: str):
    pkl_path = MODELS_DIR / f"{model_name}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Model not found: {pkl_path}\n"
            f"Available models: {[p.stem for p in MODELS_DIR.glob('*.pkl')]}"
        )
    with open(pkl_path, "rb") as f:
        model = pickle.load(f)
    log.info(f"Loaded model: {pkl_path}")
    return model


def prepare_features(df: pd.DataFrame, features: list, config: dict) -> pd.DataFrame:
    """Prepare feature matrix X — mirrors train.py's prepare_xy (X part only)."""
    leakage = set(config["leakage_columns"])
    safe_features = [f for f in features if f not in leakage and f in df.columns]

    missing = set(features) - set(safe_features) - leakage
    if missing:
        log.warning(f"Features missing from input data: {missing}")

    X = df[safe_features].copy()

    cat_cols = [c for c in ["termination", "eco_family", "source"] if c in X.columns]
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    for col in X.select_dtypes(include=np.number).columns:
        if X[col].isna().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            log.info(f"  Imputed {col} NaNs with median={median_val:.2f}")

    return X


def get_test_split(config: dict):
    """Reproduce the exact same test split used in training."""
    df = pd.read_csv(DATA_PATH, low_memory=False)
    y = df[config["target"]].astype(str)

    rs = config["random_state"]
    test_size = config["test_size"]

    _, df_test, _, y_test = train_test_split(
        df, y, test_size=test_size, random_state=rs, stratify=y
    )
    return df_test.reset_index(drop=True), y_test.reset_index(drop=True)


def predict(model_name: str, input_path: str = None):
    config = load_config()
    model = load_model(model_name)

    track = "track_b" if "track_b" in model_name else "track_a"
    features = config[f"{track}_features"]
    is_xgb = "xgboost" in model_name

    if input_path:
        df = pd.read_csv(input_path, low_memory=False)
        y_true = (
            df[config["target"]].astype(str) if config["target"] in df.columns else None
        )
        log.info(f"Predicting on custom input: {input_path} ({len(df):,} rows)")
    else:
        df, y_true = get_test_split(config)
        log.info(f"Predicting on held-out test set ({len(df):,} rows)")

    X = prepare_features(df, features, config)
    log.info(f"Feature matrix shape: {X.shape}")

    if is_xgb:
        # XGBoost was trained with alphabetically encoded labels
        le = LabelEncoder()
        le.fit(["Advanced", "Beginner", "Expert", "Intermediate", "Master"])
        y_pred_enc = model.predict(X)
        y_pred = le.inverse_transform(y_pred_enc)
    else:
        y_pred = model.predict(X)

    out_df = df.copy()
    out_df["predicted_skill"] = y_pred
    if y_true is not None:
        out_df["actual_skill"] = y_true.values
        out_df["correct"] = out_df["predicted_skill"] == out_df["actual_skill"]

    out_path = REPORTS_DIR / f"predictions_{model_name}.csv"
    out_df.to_csv(out_path, index=False)
    log.info(f"Predictions saved to {out_path}")

    if y_true is not None:
        print(f"\n{'='*60}")
        print(f"  MODEL: {model_name}")
        print(f"{'='*60}")

        print("\n── Per-Class Report ──────────────────────────────────────")
        # labels= ensures correct ordering without remapping names
        print(
            classification_report(y_true, y_pred, labels=CLASS_ORDER, zero_division=0)
        )

        print("── Confusion Matrix ──────────────────────────────────────")
        cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
        cm_df = pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER)
        cm_df.index.name = "Actual \\ Predicted"
        print(cm_df.to_string())

        print("\n── Summary ───────────────────────────────────────────────")
        correct = (np.array(y_pred) == np.array(y_true)).sum()
        total = len(y_true)
        print(f"  Overall accuracy : {correct/total:.2%} ({correct}/{total})")
        print(f"  Predictions CSV  : {out_path}")

    return y_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run predictions with a trained chess skill model"
    )
    parser.add_argument(
        "--model",
        default="random_forest_track_b",
        help="Model name to load from models/ dir (default: random_forest_track_b)",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to a custom CSV file. If omitted, uses the held-out test set.",
    )
    args = parser.parse_args()
    predict(model_name=args.model, input_path=args.input)
