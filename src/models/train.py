"""
src/models/train.py
Model training and MLflow experiment logging for chess skill classification.

Functions
---------
log_model_run  — train a sklearn model, evaluate, and log to MLflow
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

TARGET_NAMES = ["Beginner", "Intermediate", "Advanced", "Expert", "Master"]


def log_model_run(
    model_name: str,
    model,
    params: dict,
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
):
    """
    Train a sklearn model, evaluate it on val and test sets,
    and log all metrics, parameters, and artifacts to MLflow.

    Standard metrics logged
    -----------------------
    train_accuracy, val_accuracy, test_accuracy, val_macro_f1, test_macro_f1

    Business metrics logged
    -----------------------
    expert_recall    — Expert is the most populous class; misclassification
                       has the widest matchmaking impact.
    master_precision — Falsely labeling a player as Master produces the
                       worst user experience.

    Parameters
    ----------
    model_name : str        — display name for the MLflow run
    model      : estimator  — untrained sklearn estimator
    params     : dict       — hyperparameters to log
    X_train / X_val / X_test : feature matrices (scaled)
    y_train / y_val / y_test : ordinal target (0=Beginner … 4=Master)

    Returns
    -------
    Trained model
    """
    with mlflow.start_run(run_name=model_name):

        # ── Log hyperparameters ───────────────────────────────────────────────
        mlflow.log_param("model_name", model_name)
        for k, v in params.items():
            mlflow.log_param(k, v)

        # ── Train ─────────────────────────────────────────────────────────────
        model.fit(X_train, y_train)

        # ── Predict ───────────────────────────────────────────────────────────
        y_pred_train = model.predict(X_train)
        y_pred_val   = model.predict(X_val)
        y_pred_test  = model.predict(X_test)

        # ── Standard metrics ──────────────────────────────────────────────────
        train_acc     = accuracy_score(y_train, y_pred_train)
        val_acc       = accuracy_score(y_val,   y_pred_val)
        test_acc      = accuracy_score(y_test,  y_pred_test)
        val_macro_f1  = f1_score(y_val,  y_pred_val,  average="macro")
        test_macro_f1 = f1_score(y_test, y_pred_test, average="macro")

        mlflow.log_metric("train_accuracy",  train_acc)
        mlflow.log_metric("val_accuracy",    val_acc)
        mlflow.log_metric("test_accuracy",   test_acc)
        mlflow.log_metric("val_macro_f1",    val_macro_f1)
        mlflow.log_metric("test_macro_f1",   test_macro_f1)

        # ── Business metrics ──────────────────────────────────────────────────
        report = classification_report(
            y_val, y_pred_val,
            target_names=TARGET_NAMES,
            output_dict=True,
        )
        expert_recall    = report["Expert"]["recall"]
        master_precision = report["Master"]["precision"]

        mlflow.log_metric("expert_recall",    expert_recall)
        mlflow.log_metric("master_precision", master_precision)

        # ── Save model artifact ───────────────────────────────────────────────
        mlflow.sklearn.log_model(model, artifact_path="model")

        # ── Confusion matrix artifact ─────────────────────────────────────────
        cm  = confusion_matrix(y_test, y_pred_test)
        fig, ax = plt.subplots(figsize=(7, 5))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=TARGET_NAMES).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(f"{model_name} — Test Confusion Matrix")
        plt.tight_layout()
        cm_path = f"reports/figures/cm_{model_name.replace(' ', '_')}.png"
        fig.savefig(cm_path, dpi=120)
        mlflow.log_artifact(cm_path)
        plt.show()

        # ── Console summary ───────────────────────────────────────────────────
        print(f"\n{'='*55}")
        print(f"  {model_name}")
        print(f"{'='*55}")
        print(f"  Train accuracy    : {train_acc:.4f}")
        print(f"  Val   accuracy    : {val_acc:.4f}")
        print(f"  Test  accuracy    : {test_acc:.4f}")
        print(f"  Val   macro F1    : {val_macro_f1:.4f}")
        print(f"  Test  macro F1    : {test_macro_f1:.4f}")
        print(f"  Expert recall     : {expert_recall:.4f}  (business)")
        print(f"  Master precision  : {master_precision:.4f}  (business)")
        print(f"\nClassification Report (val):")
        print(classification_report(y_val, y_pred_val, target_names=TARGET_NAMES))

    return model
