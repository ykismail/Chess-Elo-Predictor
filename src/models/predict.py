"""
src/models/predict.py
Prediction utilities for the chess skill classification project.
"""

import pandas as pd
import numpy as np
import mlflow.sklearn

TARGET_NAMES = ["Beginner", "Intermediate", "Advanced", "Expert", "Master"]


def load_model(run_id: str, artifact_path: str = "model"):
    """
    Load a trained model from MLflow by run ID.

    Parameters
    ----------
    run_id        : str — MLflow run ID
    artifact_path : str — artifact path within the run (default: 'model')

    Returns
    -------
    Loaded sklearn model
    """
    model_uri = f"runs:/{run_id}/{artifact_path}"
    return mlflow.sklearn.load_model(model_uri)


def predict_skill_tier(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Predict skill tier for a set of games.

    Parameters
    ----------
    model : trained sklearn model
    X     : pd.DataFrame — preprocessed feature matrix (scaled)

    Returns
    -------
    pd.DataFrame with columns:
        predicted_enc   : int   — ordinal prediction (0=Beginner … 4=Master)
        predicted_label : str   — human-readable skill tier label
        confidence      : float — probability of predicted class (if available)
    """
    preds_enc = model.predict(X)
    preds_label = [TARGET_NAMES[p] for p in preds_enc]

    result = pd.DataFrame({
        "predicted_enc":   preds_enc,
        "predicted_label": preds_label,
    }, index=X.index)

    # Add confidence if model supports predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        result["confidence"] = proba.max(axis=1).round(4)

    return result
