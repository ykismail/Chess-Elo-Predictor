"""
train.py
========
Model training pipeline for chess skill classification.

Data context (from merged_games.csv):
- 43,694 rows, 5-class target: elo_bucket_white
- Severe class imbalance: Expert 38.9%, Beginner 0.6%
- Target is pd.Categorical created by pd.cut on white_elo
- All models use class_weight='balanced' where supported
- Hyperparameters found via GridSearchCV on training split only
"""

import json
import logging
import os
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "merged_games.csv"
CONFIG_PATH = ROOT / "configs" / "model_params.json"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports" / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Config & Data
# ─────────────────────────────────────────────────────────────────────────────


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def load_data(config: dict) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    log.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

    # Confirm target exists
    assert (
        config["target"] in df.columns
    ), f"Target '{config['target']}' not found in CSV"

    # Warn if leakage columns are still present (they should be excluded in prepare_xy)
    present_leakage = [c for c in config["leakage_columns"] if c in df.columns]
    if present_leakage:
        log.warning(
            f"Leakage columns present in CSV (will be excluded from X): {present_leakage}"
        )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────


def prepare_xy(df: pd.DataFrame, features: list, config: dict):
    """
    Build X and y from the merged DataFrame.

    Key decisions driven by data:
    - elo_bucket_white is a pd.Categorical — convert to plain string for sklearn
    - Categorical features (termination, eco_family, source) are label-encoded
    - 18 rows have NaN Stockfish values — imputed with column median
    - Leakage columns are explicitly excluded even if accidentally in features list
    """
    leakage = set(config["leakage_columns"])
    safe_features = [f for f in features if f not in leakage and f in df.columns]

    if len(safe_features) < len(features):
        dropped = set(features) - set(safe_features)
        log.warning(f"Dropped from features (leakage or missing): {dropped}")

    X = df[safe_features].copy()
    # elo_bucket_white is a pd.Categorical — sklearn needs plain strings
    y = df[config["target"]].astype(str)

    # Encode categorical columns
    cat_cols = [c for c in ["termination", "eco_family", "source"] if c in X.columns]
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    cat_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        cat_encoders[col] = le 

    # Impute numeric NaNs with median (18 Lichess games missing Stockfish)
    for col in X.select_dtypes(include=np.number).columns:
        if X[col].isna().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            log.info(f"  Imputed {col} NaNs with median={median_val:.2f}")

    log.info(f"X shape: {X.shape}, y distribution:\n{y.value_counts().to_string()}")
    return X, y, cat_encoders


def stratified_split(X, y, config: dict):
    rs = config["random_state"]
    test_size = config["test_size"]
    val_size = config["val_size"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=rs, stratify=y
    )
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_relative, random_state=rs, stratify=y_train
    )
    log.info(
        f"Split sizes — train: {len(X_train):,}  val: {len(X_val):,}  test: {len(X_test):,}"
    )
    # Log Beginner count in each split — key check given 247 total Beginner rows
    for name, y_s in [("train", y_train), ("val", y_val), ("test", y_test)]:
        beginner_n = (y_s == "Beginner").sum()
        log.info(f"  {name} Beginner count: {beginner_n}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────────────────────
# Model Definitions — justified by data characteristics
# ─────────────────────────────────────────────────────────────────────────────


def build_models(config: dict) -> dict:
    rs = config["random_state"]

    return {
        "dummy": DummyClassifier(strategy="stratified", random_state=rs),
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=rs,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced",
            random_state=rs,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            eval_metric="mlogloss",
            random_state=rs,
            n_jobs=-1,
        ),

        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                random_state=rs,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,
                max_iter=500,
            )),
        ]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter Tuning
# ─────────────────────────────────────────────────────────────────────────────


def tune_model(
    model_name: str, model, X_train, y_train, config: dict, label_encoder=None
):
    """
    Run GridSearchCV on the training split only.
    Dummy has no hyperparameters — returned as-is.
    XGBoost requires numeric labels — uses label_encoder if provided.
    """
    if model_name == "dummy":
        model.fit(X_train, y_train)
        return model, {}

    param_grid = config["param_grids"].get(model_name, {})
    if not param_grid:
        log.info(f"  No param grid for {model_name}, fitting directly.")
        train_y = label_encoder.transform(y_train) if label_encoder else y_train
        model.fit(X_train, train_y)
        return model, {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config["random_state"])

    train_y = label_encoder.transform(y_train) if label_encoder else y_train

    # ── NEW: MLP and LR need sample_weight since they can't use class_weight ──
    # This block must come BEFORE GridSearchCV is created and called
    if model_name in ("mlp", "logistic_regression"):
        sample_weight = compute_sample_weight("balanced", y_train)
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            verbose=1,
            refit=True,
        )
        # Pipeline expects clf__sample_weight not sample_weight
        search.fit(X_train, train_y, clf__sample_weight=sample_weight)
        log.info(f"  Best params for {model_name}: {search.best_params_}")
        log.info(f"  Best CV f1_macro: {search.best_score_:.4f}")
        return search.best_estimator_, search.best_params_
    # ── END NEW BLOCK ──────────────────────────────────────────────────────────

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    # XGBoost sample_weight (from earlier suggestion)
    if model_name == "xgboost":
        sample_weight = compute_sample_weight("balanced", y_train)
        search.fit(X_train, train_y, sample_weight=sample_weight)
    else:
        search.fit(X_train, train_y)

    log.info(f"  Best params for {model_name}: {search.best_params_}")
    log.info(f"  Best CV f1_macro: {search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "f1_macro":           round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "balanced_accuracy":  round(balanced_accuracy_score(y_true, y_pred), 4),   # add this
        "weighted_accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "expert_recall":      round(recall_score(y_true, y_pred, labels=["Expert"],  average="macro", zero_division=0), 4),
        "beginner_recall":    round(recall_score(y_true, y_pred, labels=["Beginner"], average="macro", zero_division=0), 4),  # track the problem class directly
        "master_precision":   round(precision_score(y_true, y_pred, labels=["Master"], average="macro", zero_division=0), 4),
    }


def compute_metrics_encoded(y_true_enc, y_pred_enc, label_encoder) -> dict:
    """Decode numeric XGBoost predictions back to string labels before computing metrics."""
    y_true = label_encoder.inverse_transform(y_true_enc)
    y_pred = label_encoder.inverse_transform(y_pred_enc)
    return compute_metrics(y_true, y_pred)


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────


def train_all(config: dict) -> pd.DataFrame:
    df = load_data(config)

    tracks = {
        "track_a": config["track_a_features"],
        "track_b": config["track_b_features"],
    }

    all_results = []
    mlflow.set_experiment("chess_skill_classification")

    for track_name, features in tracks.items():
        log.info(
            f"\n{'='*60}\nTRACK: {track_name.upper()} ({len(features)} features)\n{'='*60}"
        )

        X, y, cat_encoders = prepare_xy(df, features, config)  # modify prepare_xy to return encoders
        # save encoders per track
        encoders_path = MODELS_DIR / f"cat_encoders_{track_name}.pkl"
        with open(encoders_path, "wb") as f:
            pickle.dump(cat_encoders, f)
        X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y, config)

        # after stratified_split call:
        sm = SMOTE(random_state=config["random_state"], k_neighbors=5)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        log.info(f"After SMOTE — train shape: {X_train.shape}, y distribution:\n{pd.Series(y_train).value_counts().to_string()}")

        # Fit a global label encoder for XGBoost (needs numeric labels)
        global_le = LabelEncoder()
        global_le.fit(y)

        models = build_models(config)

        for model_name, model in models.items():
            log.info(f"\n  ── {model_name.upper()} ──")

            # XGBoost requires numeric labels
            is_xgb = model_name == "xgboost"
            le = global_le if is_xgb else None

            with mlflow.start_run(run_name=f"{model_name}_{track_name}"):

                best_model, best_params = tune_model(
                    model_name, model, X_train, y_train, config, label_encoder=le
                )

                mlflow.log_param("model", model_name)
                mlflow.log_param("track", track_name)
                mlflow.log_param("n_features", len(features))
                mlflow.log_param("n_train", len(X_train))
                mlflow.log_param(
                    "class_imbalance_handling",
                    "class_weight=balanced + stratified_kfold",
                )
                for k, v in best_params.items():
                    mlflow.log_param(k, v)

                for split_name, X_s, y_s in [
                    ("train", X_train, y_train),
                    ("val", X_val, y_val),
                    ("test", X_test, y_test),
                ]:
                    if is_xgb:
                        y_s_enc = le.transform(y_s)
                        y_pred = best_model.predict(X_s)
                        metrics = compute_metrics_encoded(y_s_enc, y_pred, le)
                    else:
                        y_pred = best_model.predict(X_s)
                        metrics = compute_metrics(y_s, y_pred)

                    for k, v in metrics.items():
                        mlflow.log_metric(f"{split_name}_{k}", v)

                    if split_name == "test":
                        log.info(f"  TEST → {metrics}")
                        all_results.append(
                            {
                                "model": model_name,
                                "track": track_name,
                                **{f"test_{k}": v for k, v in metrics.items()},
                            }
                        )

                # Classification report
                if is_xgb:
                    y_pred_test = best_model.predict(X_test)
                    y_pred_test_str = le.inverse_transform(y_pred_test)
                    report = classification_report(
                        y_test,
                        y_pred_test_str,
                        target_names=config["class_order"],
                        zero_division=0,
                    )
                else:
                    y_pred_test = best_model.predict(X_test)
                    report = classification_report(
                        y_test,
                        y_pred_test,
                        target_names=config["class_order"],
                        zero_division=0,
                    )

                report_path = REPORTS_DIR / f"{model_name}_{track_name}_report.txt"
                report_path.write_text(report)
                mlflow.log_artifact(str(report_path))

                mlflow.sklearn.log_model(
                    best_model, artifact_path=f"{model_name}_{track_name}"
                )
                pkl_path = MODELS_DIR / f"{model_name}_{track_name}.pkl"
                with open(pkl_path, "wb") as f:
                    pickle.dump(best_model, f)
                log.info(f"  Saved model to {pkl_path}")

    results_df = pd.DataFrame(all_results)
    results_path = REPORTS_DIR / "model_comparison.csv"
    results_df.to_csv(results_path, index=False)
    log.info(f"\nComparison table:\n{results_df.to_string(index=False)}")
    return results_df


if __name__ == "__main__":
    cfg = load_config()
    train_all(cfg)
