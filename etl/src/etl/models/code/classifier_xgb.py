"""
XGBoost-based page classifier training script.

This module trains an XGBoost model for initial page classification using
hand-crafted features extracted from text and HTML content.
"""

import json
from typing import cast

import numpy as np
import optuna
from optuna.integration import XGBoostPruningCallback
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

try:
    from .classifier_utils import extract_features
    from .shared_constants import CLASSIFIER_LABEL_LIST, CLASSIFIER_XGB_PATH
except ImportError:  # pragma: no cover - supports running as a script
    from classifier_utils import extract_features  # pyright: ignore[reportMissingImports]
    from shared_constants import (  # pyright: ignore[reportMissingImports]
        CLASSIFIER_LABEL_LIST,
        CLASSIFIER_XGB_PATH,
    )


def load_and_prepare_data(
    data_path: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
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
    y = df["label"].map(label2idx).to_numpy()

    # Build feature matrix using parallel processing
    print(f"[data] extracting features in parallel...")
    features_list = cast(
        list[np.ndarray],
        list(
            Parallel(n_jobs=-1, verbose=1)(
                delayed(extract_features)(row["text"], row["html"], row["order"])
                for _, row in df.iterrows()
            )
        ),
    )
    features = np.vstack(features_list)

    return features, y, label2idx


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


def objective(
    trial: optuna.Trial, dtrain: xgb.DMatrix, dval: xgb.DMatrix, y_val: np.ndarray
) -> float:
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
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        "max_bin": trial.suggest_categorical("max_bin", [128, 256, 384]),
    }

    pruning_callback = XGBoostPruningCallback(trial, "val-f1_macro")

    # Train model
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "val")],
        custom_metric=macro_f1_eval,
        maximize=True,
        early_stopping_rounds=50,
        verbose_eval=False,
        callbacks=[pruning_callback],
    )

    best_it = getattr(model, "best_iteration", None)
    if best_it is None:
        best_it = getattr(model, "num_boosted_rounds", lambda: 500)()
    trial.set_user_attr("best_iteration", int(best_it))

    # Evaluate
    y_prob = model.predict(dval)
    y_pred = np.argmax(y_prob, axis=1)
    f1 = f1_score(y_val, y_pred, average="macro")

    return f1


def _nll_with_temperature(
    probs: np.ndarray, y_true: np.ndarray, temperature: float
) -> float:
    # probs shape: (N, C), valid probs in (0,1); y_true: (N,)
    p = np.clip(probs, 1e-12, 1.0)
    # treat log-probs as logits; apply temperature scaling in log space
    logits = np.log(p)  # already clipped away from 0
    scaled_logits = logits / temperature
    # subtract max for numerical stability
    logits_max = scaled_logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(scaled_logits - logits_max)
    # normalize
    probs_T = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    # compute NLL
    idx = (np.arange(y_true.size), y_true)
    return float(-np.log(probs_T[idx]).mean())


def _fit_temperature(probs: np.ndarray, y_true: np.ndarray) -> float:
    # simple coarse-to-fine search over T in [0.5, 3.0]
    grid = np.linspace(0.5, 3.0, 26)
    bestT, bestLoss = 1.0, 1e18
    for temperature in grid:
        loss = _nll_with_temperature(probs, y_true, temperature)
        if loss < bestLoss:
            bestLoss, bestT = loss, temperature
    # local refine around bestT
    for step in [0.2, 0.1, 0.05]:
        lo, hi = max(0.1, bestT - step), bestT + step
        cand = np.linspace(lo, hi, 11)
        for temperature in cand:
            loss = _nll_with_temperature(probs, y_true, temperature)
            if loss < bestLoss:
                bestLoss, bestT = loss, temperature
    return float(bestT)


def main() -> None:
    """Main training function for XGBoost classifier."""
    # Load and prepare data
    data_path = "etl/src/etl/models/data/page-data.parquet"
    features, y, label2idx = load_and_prepare_data(data_path)
    _ = label2idx
    features = features.astype(np.float32, copy=False)

    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(
        features, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train = np.asarray(X_train)
    X_val = np.asarray(X_val)
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)

    # Split validation into model-validation and calibration
    X_val_model, X_cal, y_val_model, y_cal = train_test_split(
        X_val, y_val, test_size=0.2, random_state=42, stratify=y_val
    )
    X_val_model = np.asarray(X_val_model)
    X_cal = np.asarray(X_cal)
    y_val_model = np.asarray(y_val_model)
    y_cal = np.asarray(y_cal)
    dval_model = xgb.DMatrix(X_val_model, label=y_val_model)
    dcal = xgb.DMatrix(X_cal, label=y_cal)

    # Create XGBoost datasets (weights from TRAIN ONLY)
    num_classes = len(CLASSIFIER_LABEL_LIST)
    counts_train = np.bincount(y_train, minlength=num_classes).astype(float)
    class_w = counts_train.sum() / np.maximum(counts_train, 1.0)
    class_w = class_w * (num_classes / class_w.sum())  # mean ≈ 1
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=class_w[y_train])

    # Hyperparameter optimization
    print("Starting hyperparameter optimization...")
    study = optuna.create_study(direction="maximize")

    # Define objective with fixed data
    def objective_wrapper(trial: optuna.Trial) -> float:
        """Wrapper for objective function with fixed data."""
        return objective(trial, dtrain, dval_model, y_val_model)

    study.optimize(objective_wrapper, n_trials=500)
    print("Best hyperparameters:", study.best_params)

    best_it = int(study.best_trial.user_attrs.get("best_iteration", 500))
    best_params = study.best_params.copy()
    best_params.update(
        {
            "objective": "multi:softprob",
            "num_class": len(CLASSIFIER_LABEL_LIST),
            "eval_metric": "mlogloss",
            "tree_method": "hist",
        }
    )

    final_model = xgb.train(best_params, dtrain, num_boost_round=best_it)
    final_model.save_model(CLASSIFIER_XGB_PATH)
    print(f"Model saved to {CLASSIFIER_XGB_PATH}")

    # ---- Temperature scaling on calibration slice; save calibrator ----
    y_prob_calib = final_model.predict(dcal)  # probs on calib set
    temperature = _fit_temperature(y_prob_calib, y_cal)
    calib_path = CLASSIFIER_XGB_PATH + ".calib.json"
    with open(calib_path, "w") as f:
        json.dump({"temperature": float(temperature)}, f)
    print(f"Saved temperature T={temperature:.3f} to {calib_path}")

    # Evaluate on MODEL-VAL (raw & calibrated)
    def _apply_T(probs: np.ndarray, temperature: float) -> np.ndarray:
        p = np.clip(probs, 1e-12, 1.0)
        logp = np.log(p) / temperature
        logp = logp - np.logaddexp.reduce(logp, axis=1, keepdims=True)
        return np.exp(logp)

    y_prob_model = final_model.predict(dval_model)
    y_hat_raw = np.argmax(y_prob_model, axis=1)
    accuracy = accuracy_score(y_val_model, y_hat_raw)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val_model, y_hat_raw, average="macro"
    )
    print(
        f"Model-Val (raw)  Acc: {accuracy:.4f}  P/R/F1: {precision:.4f}/{recall:.4f}/{f1:.4f}"
    )
    cm_raw = confusion_matrix(y_val_model, y_hat_raw, labels=np.arange(num_classes))
    print("Model-Val (raw) Confusion Matrix:")
    print(cm_raw)
    class_acc = cm_raw.diagonal() / np.maximum(cm_raw.sum(axis=1), 1)
    per_prec_arr, per_rec_arr, per_f1_arr, _ = precision_recall_fscore_support(
        y_val_model, y_hat_raw, labels=np.arange(num_classes), average=None
    )
    per_prec = np.asarray(per_prec_arr)
    per_rec = np.asarray(per_rec_arr)
    per_f1 = np.asarray(per_f1_arr)
    print("Model-Val (raw) Per-class metrics:")
    for i, label in enumerate(CLASSIFIER_LABEL_LIST):
        print(
            f"  {label}: Acc={class_acc[i]:.4f} "
            f"P={per_prec[i]:.4f} R={per_rec[i]:.4f} F1={per_f1[i]:.4f}"
        )

    y_prob_model_cal = _apply_T(y_prob_model, temperature)
    y_hat_cal = np.argmax(y_prob_model_cal, axis=1)
    precision_c, recall_c, f1_c, _ = precision_recall_fscore_support(
        y_val_model, y_hat_cal, average="macro"
    )
    print(f"Model-Val (cal)   P/R/F1: {precision_c:.4f}/{recall_c:.4f}/{f1_c:.4f}")
    cm_cal = confusion_matrix(y_val_model, y_hat_cal, labels=np.arange(num_classes))
    print("Model-Val (cal) Confusion Matrix:")
    print(cm_cal)
    class_acc_c = cm_cal.diagonal() / np.maximum(cm_cal.sum(axis=1), 1)
    per_prec_c_arr, per_rec_c_arr, per_f1_c_arr, _ = precision_recall_fscore_support(
        y_val_model, y_hat_cal, labels=np.arange(num_classes), average=None
    )
    per_prec_c = np.asarray(per_prec_c_arr)
    per_rec_c = np.asarray(per_rec_c_arr)
    per_f1_c = np.asarray(per_f1_c_arr)
    print("Model-Val (cal) Per-class metrics:")
    for i, label in enumerate(CLASSIFIER_LABEL_LIST):
        print(
            f"  {label}: Acc={class_acc_c[i]:.4f} "
            f"P={per_prec_c[i]:.4f} R={per_rec_c[i]:.4f} F1={per_f1_c[i]:.4f}"
        )


if __name__ == "__main__":
    main()
