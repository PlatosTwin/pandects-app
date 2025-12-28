"""
XGBoost-based page classifier training script.

This module trains an XGBoost model for initial page classification using
hand-crafted features extracted from text and HTML content. Agreement splits
are loaded from a shared manifest to prevent leakage.
"""
# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false

import os
from collections.abc import Sequence
from typing import cast

import numpy as np
import optuna
from optuna.integration import XGBoostPruningCallback
from optuna.samplers import TPESampler
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from joblib import Parallel, delayed
import yaml

try:
    from .classifier_utils import extract_features
    from .shared_constants import CLASSIFIER_LABEL_LIST, CLASSIFIER_XGB_PATH
    from .split_utils import (
        build_agreement_split,
        load_split_manifest,
        write_split_manifest,
    )
except ImportError:  # pragma: no cover - supports running as a script
    from classifier_utils import extract_features  # pyright: ignore[reportMissingImports]
    from shared_constants import (  # pyright: ignore[reportMissingImports]
        CLASSIFIER_LABEL_LIST,
        CLASSIFIER_XGB_PATH,
    )
    from split_utils import (  # pyright: ignore[reportMissingImports]
        build_agreement_split,
        load_split_manifest,
        write_split_manifest,
    )

CODE_DIR = os.path.dirname(__file__)
EVAL_METRICS_DIR = os.path.normpath(os.path.join(CODE_DIR, "../eval_metrics"))
SEED = 42


def load_and_prepare_data(
    data_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load and prepare data for XGBoost training.

    Args:
        data_path: Path to the parquet file containing page data

    Returns:
        Tuple of (features, labels, agreement_uuids, announcement_years, split_df)
    """
    df = pd.read_parquet(data_path)

    required_cols = {
        "html",
        "text",
        "label",
        "order",
        "date_announcement",
        "agreement_uuid",
    }
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Data must contain columns: {sorted(required_cols)}. Missing: {sorted(missing)}")

    print(f"[data] loaded {df.shape[0]} rows from {data_path}")

    # Parse announcement year for stratified splits
    announcement_dates = pd.to_datetime(df["date_announcement"], errors="raise")
    if announcement_dates.isna().any():
        raise ValueError("Found missing or invalid date_announcement values.")
    announcement_years = announcement_dates.dt.year.to_numpy()
    agreement_uuids = df["agreement_uuid"].astype(str).to_numpy()
    year_counts = (
        pd.DataFrame(
            {"agreement_uuid": agreement_uuids, "announcement_year": announcement_years}
        )
        .groupby("agreement_uuid")["announcement_year"]
        .nunique()
        .astype(int)
    )
    inconsistent = year_counts[year_counts > 1]
    if not inconsistent.empty:
        raise ValueError(
            "Found agreements spanning multiple announcement years; cannot stratify by year."
        )

    # Map labels to integers
    labels = cast(list[str], CLASSIFIER_LABEL_LIST)
    label2idx = {label: idx for idx, label in enumerate(labels)}
    y_series = df["label"].map(label2idx)
    if y_series.isna().any():
        raise ValueError("Found labels missing from CLASSIFIER_LABEL_LIST.")
    y = y_series.astype(int).to_numpy()

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

    split_df = df[["agreement_uuid", "date_announcement", "label"]].copy()
    return features, y, agreement_uuids, announcement_years, split_df


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
        "seed": SEED,
        "nthread": 1,
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


def main() -> None:
    """Main training function for XGBoost classifier."""
    np.random.seed(SEED)
    year_window = 5
    length_bucket_edges = [0.0, 120, 130, 200, float("inf")]
    back_matter_bucket_edges = [0.0, 35, 60, 105, float("inf")]
    # Load and prepare data
    data_path = "etl/src/etl/models/data/page-data.parquet"
    features, y, agreement_uuids, years, split_df = load_and_prepare_data(
        data_path
    )
    split_path = "etl/src/etl/models/data/agreement-splits.json"
    if os.path.exists(split_path):
        split = load_split_manifest(split_path)
    else:
        split = build_agreement_split(
            split_df,
            val_split=0.1,
            test_split=0.1,
            year_window=year_window,
            length_bucket_edges=length_bucket_edges,
            back_matter_bucket_edges=back_matter_bucket_edges,
        )
        _ = write_split_manifest(split_path, split)
        print(f"[split] wrote agreement split manifest to {split_path}")
    features = features.astype(np.float32, copy=False)

    # Train/val/test split from shared manifest (agreement-level, year-stratified)
    agreement_ids = pd.unique(agreement_uuids).astype(str)
    if "train" not in split or "val" not in split or "test" not in split:
        raise ValueError("Split manifest missing required keys: train/val/test.")
    train_ids = [str(x) for x in split["train"]]
    val_ids = [str(x) for x in split["val"]]
    test_ids = [str(x) for x in split["test"]]

    df_agreement_ids = {str(x) for x in agreement_ids}
    split_ids = set(train_ids) | set(val_ids) | set(test_ids)
    missing_ids = split_ids - df_agreement_ids
    if missing_ids:
        raise ValueError("Split manifest contains unknown agreement_uuid values.")

    agreement_windows = (
        pd.DataFrame(
            {"agreement_uuid": agreement_uuids, "announcement_year": years}
        )
        .drop_duplicates("agreement_uuid")
        .set_index("agreement_uuid")["announcement_year"]
        .loc[agreement_ids]
    )
    agreement_window_map = (
        (agreement_windows.astype(int) // year_window) * year_window
    ).astype(int)

    train_mask = np.isin(agreement_uuids, train_ids)
    val_model_mask = np.isin(agreement_uuids, val_ids)
    test_mask = np.isin(agreement_uuids, test_ids)

    split_meta = split.get("meta")
    length_edges_raw = split_meta["length_bucket_edges"] if split_meta else None
    back_edges_raw = split_meta["back_matter_bucket_edges"] if split_meta else None
    length_edges = (
        [float(x) for x in length_edges_raw]
        if isinstance(length_edges_raw, list)
        else None
    )
    back_edges = (
        [float(x) for x in back_edges_raw]
        if isinstance(back_edges_raw, list)
        else None
    )
    agreement_stats = (
        split_df.assign(
            agreement_uuid=split_df["agreement_uuid"].astype(str),
            announcement_year=pd.to_datetime(
                split_df["date_announcement"], errors="raise"
            ).dt.year,
            is_back=split_df["label"].astype(str).eq("back_matter"),
        )
        .groupby("agreement_uuid")
        .agg(
            page_count=("agreement_uuid", "size"),
            back_count=("is_back", "sum"),
            announcement_year=("announcement_year", "first"),
        )
    )
    agreement_stats["announcement_window"] = (
        (agreement_stats["announcement_year"].astype(int) // year_window) * year_window
    )

    def _bucket_counts(values: pd.Series, edges: list[float]) -> dict[int, int]:
        buckets = pd.cut(values, bins=edges, include_lowest=True, labels=False)
        if buckets.isna().any():
            raise ValueError("Failed to bucketize split metric values.")
        return buckets.astype(int).value_counts().sort_index().to_dict()

    def _split_report(name: str, ids: Sequence[str]) -> dict[str, object]:
        split_pages = np.isin(agreement_uuids, list(ids))
        page_count = int(split_pages.sum())
        window_counts = (
            agreement_window_map.loc[list(ids)]
            .astype(int)
            .value_counts()
            .sort_index()
            .to_dict()
        )
        length_counts: dict[int, int] | None = None
        back_counts: dict[int, int] | None = None
        if length_edges:
            length_counts = _bucket_counts(
                agreement_stats.loc[list(ids), "page_count"], length_edges
            )
        if back_edges:
            back_counts = _bucket_counts(
                agreement_stats.loc[list(ids), "back_count"], back_edges
            )
        print(
            f"[split] {name:<8} agreements={len(ids):>4} pages={page_count:>6} windows={window_counts}"
        )
        if length_counts is not None:
            print(f"[split] {name:<8} length_bucket_counts={length_counts}")
        if back_counts is not None:
            print(f"[split] {name:<8} back_bucket_counts={back_counts}")
        return {
            "agreements": int(len(ids)),
            "pages": page_count,
            "announcement_window_counts": {int(k): int(v) for k, v in window_counts.items()},
            "length_bucket_counts": (
                {int(k): int(v) for k, v in length_counts.items()} if length_counts is not None else None
            ),
            "back_bucket_counts": (
                {int(k): int(v) for k, v in back_counts.items()} if back_counts is not None else None
            ),
        }

    print(f"[split] using agreement manifest: {split_path}")
    split_reports = {
        "train": _split_report("train", train_ids),
        "val": _split_report("val", val_ids),
        "test": _split_report("test", test_ids),
    }

    X_train = np.asarray(features[train_mask])
    y_train = np.asarray(y[train_mask])
    X_val_model = np.asarray(features[val_model_mask])
    y_val_model = np.asarray(y[val_model_mask])
    X_test = np.asarray(features[test_mask])
    y_test = np.asarray(y[test_mask])
    dval_model = xgb.DMatrix(X_val_model, label=y_val_model)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Create XGBoost datasets (weights from TRAIN ONLY)
    num_classes = len(CLASSIFIER_LABEL_LIST)
    counts_train = np.bincount(y_train, minlength=num_classes).astype(float)
    class_w = counts_train.sum() / np.maximum(counts_train, 1.0)
    class_w = class_w * (num_classes / class_w.sum())  # mean ≈ 1
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=class_w[y_train])

    # Hyperparameter optimization
    print("Starting hyperparameter optimization...")
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))

    # Define objective with fixed data
    def objective_wrapper(trial: optuna.Trial) -> float:
        """Wrapper for objective function with fixed data."""
        return objective(trial, dtrain, dval_model, y_val_model)

    study.optimize(objective_wrapper, n_trials=600)
    print("Best hyperparameters:", study.best_params)

    best_it = int(study.best_trial.user_attrs.get("best_iteration", 600))
    best_params = study.best_params.copy()
    best_params.update(
        {
            "objective": "multi:softprob",
            "num_class": len(CLASSIFIER_LABEL_LIST),
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "seed": SEED,
            "nthread": 1,
        }
    )

    train_full_mask = train_mask | val_model_mask
    X_train_full = np.asarray(features[train_full_mask])
    y_train_full = np.asarray(y[train_full_mask])
    counts_train_full = np.bincount(y_train_full, minlength=num_classes).astype(float)
    class_w_full = counts_train_full.sum() / np.maximum(counts_train_full, 1.0)
    class_w_full = class_w_full * (num_classes / class_w_full.sum())
    dtrain_full = xgb.DMatrix(
        X_train_full, label=y_train_full, weight=class_w_full[y_train_full]
    )

    final_model = xgb.train(best_params, dtrain_full, num_boost_round=best_it)
    model_path = cast(str, CLASSIFIER_XGB_PATH)
    final_model.save_model(model_path)
    print(f"Model saved to {model_path}")

    # Evaluate on TEST
    y_prob_test = final_model.predict(dtest)
    y_hat_raw = np.argmax(y_prob_test, axis=1)
    accuracy = accuracy_score(y_test, y_hat_raw)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_hat_raw, average="macro"
    )
    print(
        f"Test (raw)  Acc: {accuracy:.4f}  P/R/F1: {precision:.4f}/{recall:.4f}/{f1:.4f}"
    )
    cm_raw = confusion_matrix(y_test, y_hat_raw, labels=np.arange(num_classes))
    print("Test (raw) Confusion Matrix:")
    print(cm_raw)
    class_acc = cm_raw.diagonal() / np.maximum(cm_raw.sum(axis=1), 1)
    per_prec_arr, per_rec_arr, per_f1_arr, _ = precision_recall_fscore_support(
        y_test, y_hat_raw, labels=np.arange(num_classes), average=None
    )
    per_prec = np.asarray(per_prec_arr)
    per_rec = np.asarray(per_rec_arr)
    per_f1 = np.asarray(per_f1_arr)
    print("Test (raw) Per-class metrics:")
    for i, label in enumerate(CLASSIFIER_LABEL_LIST):
        print(
            f"  {label}: Acc={class_acc[i]:.4f} P={per_prec[i]:.4f} R={per_rec[i]:.4f} F1={per_f1[i]:.4f}"
        )

    metrics = {
        "split": {
            "path": split_path,
            "year_window": int(year_window),
            "length_bucket_edges": length_edges,
            "back_matter_bucket_edges": back_edges,
            "summary": split_reports,
        },
        "overall": {
            "accuracy": float(accuracy),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
        },
        "confusion_matrix": cm_raw.astype(int).tolist(),
        "per_class": {
            label: {
                "accuracy": float(class_acc[i]),
                "precision": float(per_prec[i]),
                "recall": float(per_rec[i]),
                "f1": float(per_f1[i]),
            }
            for i, label in enumerate(CLASSIFIER_LABEL_LIST)
        },
    }
    os.makedirs(EVAL_METRICS_DIR, exist_ok=True)
    metrics_path = os.path.join(EVAL_METRICS_DIR, "classifier_xgb_test_metrics.yaml")
    with open(metrics_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(metrics, f, sort_keys=False)
    print(f"Test metrics written to {metrics_path}")

if __name__ == "__main__":
    main()
