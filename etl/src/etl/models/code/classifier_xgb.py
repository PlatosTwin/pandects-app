"""
XGBoost-based page classifier training script.

This module trains an XGBoost model for initial page classification using
hand-crafted features extracted from text and HTML content.
"""

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from .classifier_utils import extract_features
from .shared_constants import CLASSIFIER_XGB_PATH, CLASSIFIER_LABEL_LIST


def load_and_prepare_data(data_path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load and prepare data for XGBoost training.
    
    Args:
        data_path: Path to the parquet file containing page data
        
    Returns:
        Tuple of (features, labels, label_mapping)
    """
    df = pd.read_parquet(data_path)
    
    if not {"html", "text", "label"}.issubset(df.columns):
        raise ValueError("Data must contain 'html', 'text', and 'label' columns")
    
    print(f"[data] loaded {df.shape[0]} rows from {data_path}")

    # Map labels to integers
    labels = CLASSIFIER_LABEL_LIST
    label2idx = {label: idx for idx, label in enumerate(labels)}
    y = df["label"].map(label2idx).values

    # Build feature matrix
    X = np.vstack(
        [extract_features(row["text"], row["html"], row["order"]) for _, row in df.iterrows()]
    )

    return X, y, label2idx


def macro_f1_eval(preds: np.ndarray, dmatrix: xgb.DMatrix) -> tuple[str, float]:
    """
    Custom evaluation function for XGBoost using macro F1 score.
    
    Args:
        preds: Raw predictions from XGBoost
        dmatrix: XGBoost DMatrix containing true labels
        
    Returns:
        Tuple of (metric_name, metric_value)
    """
    # Get true labels
    labels = dmatrix.get_label().astype(int)
    
    # Reshape predictions
    num_samples = labels.shape[0]
    num_classes = int(preds.size / num_samples)
    preds = preds.reshape(num_samples, num_classes)
    
    # Get predicted classes
    y_pred = preds.argmax(axis=1)
    
    return "f1_macro", f1_score(labels, y_pred, average="macro")


def objective(trial: optuna.Trial, dtrain: xgb.DMatrix, dval: xgb.DMatrix, y_val: np.ndarray) -> float:
    """
    Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        dtrain: Training data matrix
        dval: Validation data matrix
        y_val: Validation labels
        
    Returns:
        Validation F1 score
    """
    # Define hyperparameter search space
    params = {
        "objective": "multi:softprob",
        "num_class": len(CLASSIFIER_LABEL_LIST),
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "eta": trial.suggest_float("eta", 1e-3, 1e-1, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }
    
    # Train model
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dval, "val")],
        custom_metric=macro_f1_eval,
        early_stopping_rounds=20,
        verbose_eval=False,
    )

    # Evaluate
    y_prob = model.predict(dval)
    y_pred = np.argmax(y_prob, axis=1)
    f1 = f1_score(y_val, y_pred, average="macro")

    return f1


def main():
    """Main training function for XGBoost classifier."""
    # Load and prepare data
    data_path = "etl/src/etl/models/data/page-data.parquet"
    X, y, label2idx = load_and_prepare_data(data_path)

    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create XGBoost datasets
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Hyperparameter optimization
    print("Starting hyperparameter optimization...")
    study = optuna.create_study(direction="maximize")
    
    # Define objective with fixed data
    def objective_wrapper(trial):
        return objective(trial, dtrain, dval, y_val)
    
    study.optimize(objective_wrapper, n_trials=400)

    print("Best hyperparameters:", study.best_params)

    # Train final model with best parameters
    best_params = study.best_params.copy()
    best_params.update({
        "objective": "multi:softprob",
        "num_class": len(CLASSIFIER_LABEL_LIST),
        "eval_metric": "mlogloss",
        "tree_method": "hist",
    })
    
    final_model = xgb.train(best_params, dtrain, num_boost_round=study.best_trial.number)

    # Save model
    final_model.save_model(CLASSIFIER_XGB_PATH)
    print(f"Model saved to {CLASSIFIER_XGB_PATH}")

    # Evaluate final model
    y_hat = np.argmax(final_model.predict(dval), axis=1)
    accuracy = accuracy_score(y_val, y_hat)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_hat, average="macro")
    
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


if __name__ == "__main__":
    main()
