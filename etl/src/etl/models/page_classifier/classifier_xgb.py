"""
XGBoost-based page classifier training script.

This module trains an XGBoost model for initial page classification using
hand-crafted features extracted from text and HTML content. Agreement splits
are loaded from a shared manifest to prevent leakage.
"""
# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

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
import yaml

if TYPE_CHECKING:
    from .classifier_utils import (
        build_feature_matrix,
        hard_negative_back_matter_mask,
    )
    from .page_classifier_constants import (
        CLASSIFIER_LABEL_LIST,
        CLASSIFIER_XGB_MOE_BASE_PATH,
        CLASSIFIER_XGB_MOE_BASE_TRAIN_PATH,
        CLASSIFIER_XGB_MOE_ROUTER_PATH,
        CLASSIFIER_XGB_MOE_ROUTER_TRAIN_PATH,
        CLASSIFIER_XGB_MOE_TAIL_PATH,
        CLASSIFIER_XGB_MOE_TAIL_TRAIN_PATH,
        CLASSIFIER_XGB_PATH,
        CLASSIFIER_XGB_TRAIN_PATH,
    )
    from .split_utils import (
        build_agreement_split,
        load_split_manifest,
        write_split_manifest,
    )
else:
    try:
        from .classifier_utils import (
            build_feature_matrix,
            hard_negative_back_matter_mask,
        )
        from .page_classifier_constants import (
            CLASSIFIER_LABEL_LIST,
            CLASSIFIER_XGB_MOE_BASE_PATH,
            CLASSIFIER_XGB_MOE_BASE_TRAIN_PATH,
            CLASSIFIER_XGB_MOE_ROUTER_PATH,
            CLASSIFIER_XGB_MOE_ROUTER_TRAIN_PATH,
            CLASSIFIER_XGB_MOE_TAIL_PATH,
            CLASSIFIER_XGB_MOE_TAIL_TRAIN_PATH,
            CLASSIFIER_XGB_PATH,
            CLASSIFIER_XGB_TRAIN_PATH,
        )
        from .split_utils import (
            build_agreement_split,
            load_split_manifest,
            write_split_manifest,
        )
    except ImportError:  # pragma: no cover - supports running as a script
        from classifier_utils import (
            build_feature_matrix,
            hard_negative_back_matter_mask,
        )
        from page_classifier_constants import (
            CLASSIFIER_LABEL_LIST,
            CLASSIFIER_XGB_MOE_BASE_PATH,
            CLASSIFIER_XGB_MOE_BASE_TRAIN_PATH,
            CLASSIFIER_XGB_MOE_ROUTER_PATH,
            CLASSIFIER_XGB_MOE_ROUTER_TRAIN_PATH,
            CLASSIFIER_XGB_MOE_TAIL_PATH,
            CLASSIFIER_XGB_MOE_TAIL_TRAIN_PATH,
            CLASSIFIER_XGB_PATH,
            CLASSIFIER_XGB_TRAIN_PATH,
        )
        from split_utils import (
            build_agreement_split,
            load_split_manifest,
            write_split_manifest,
        )

CODE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.normpath(os.path.join(CODE_DIR, "./data"))
EVAL_METRICS_DIR = os.path.normpath(os.path.join(CODE_DIR, "./eval_metrics"))
SEED = 2718
NUM_BOOST_ROUND = int(os.getenv("PAGE_CLASSIFIER_XGB_NUM_BOOST_ROUND", "2000"))
EARLY_STOPPING_ROUNDS = int(os.getenv("PAGE_CLASSIFIER_XGB_EARLY_STOPPING_ROUNDS", "50"))
OPTUNA_TRIALS = int(os.getenv("PAGE_CLASSIFIER_XGB_TRIALS", "600"))
USE_NEIGHBOR_CONTEXT = os.getenv("PAGE_CLASSIFIER_XGB_USE_NEIGHBOR_CONTEXT", "1") == "1"
USE_HARD_NEGATIVES = os.getenv("PAGE_CLASSIFIER_XGB_USE_HARD_NEGATIVES", "1") == "1"
TRAINING_MODE = os.getenv("PAGE_CLASSIFIER_XGB_MODE", "single").strip().lower()
MOE_BASE_TRIALS = int(os.getenv("PAGE_CLASSIFIER_XGB_MOE_BASE_TRIALS", str(OPTUNA_TRIALS)))
MOE_TAIL_TRIALS = int(os.getenv("PAGE_CLASSIFIER_XGB_MOE_TAIL_TRIALS", str(OPTUNA_TRIALS)))
MOE_BODY_BLEND = float(os.getenv("PAGE_CLASSIFIER_XGB_MOE_BODY_BLEND", "0.55"))
MOE_CASE_THRESHOLD = float(os.getenv("PAGE_CLASSIFIER_XGB_MOE_CASE_THRESHOLD", "0.5"))
MOE_BLEND_GRID_RAW = os.getenv(
    "PAGE_CLASSIFIER_XGB_MOE_BLEND_GRID",
    "0.35,0.45,0.55,0.65,0.75",
)
MOE_CASE_THRESHOLD_GRID_RAW = os.getenv(
    "PAGE_CLASSIFIER_XGB_MOE_CASE_THRESHOLD_GRID",
    "0.40,0.50,0.60",
)
MOE_TUNE_ON_VAL = os.getenv("PAGE_CLASSIFIER_XGB_MOE_TUNE_ON_VAL", "1") == "1"
MOE_ROUTER_NUM_BOOST_ROUND = int(
    os.getenv("PAGE_CLASSIFIER_XGB_MOE_ROUTER_NUM_BOOST_ROUND", "600")
)
MOE_ROUTER_EARLY_STOPPING_ROUNDS = int(
    os.getenv("PAGE_CLASSIFIER_XGB_MOE_ROUTER_EARLY_STOPPING_ROUNDS", "30")
)
MOE_ROUTER_TRIALS = int(os.getenv("PAGE_CLASSIFIER_XGB_MOE_ROUTER_TRIALS", "60"))


def _class_weight_vector(
    y_train: np.ndarray,
    num_classes: int,
    *,
    sig_weight_scale: float,
    back_weight_scale: float,
) -> np.ndarray:
    counts_train = np.bincount(y_train, minlength=num_classes).astype(float)
    class_w = counts_train.sum() / np.maximum(counts_train, 1.0)
    class_w = class_w * (num_classes / class_w.sum())
    sig_idx = CLASSIFIER_LABEL_LIST.index("sig")
    back_idx = CLASSIFIER_LABEL_LIST.index("back_matter")
    class_w[sig_idx] = class_w[sig_idx] * sig_weight_scale
    class_w[back_idx] = class_w[back_idx] * back_weight_scale
    class_w = class_w * (num_classes / class_w.sum())
    return class_w


def _sig_critical_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    num_classes = len(CLASSIFIER_LABEL_LIST)
    sig_idx = CLASSIFIER_LABEL_LIST.index("sig")
    back_idx = CLASSIFIER_LABEL_LIST.index("back_matter")
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    per_prec_arr, per_rec_arr, per_f1_arr, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
        average=None,
        zero_division=cast(str, cast(object, 0)),
    )
    per_prec = np.asarray(per_prec_arr)
    per_rec = np.asarray(per_rec_arr)
    per_f1 = np.asarray(per_f1_arr)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    sig_total = max(int(cm[sig_idx].sum()), 1)
    sig_to_back_rate = float(cm[sig_idx, back_idx]) / float(sig_total)
    back_total = max(int(cm[back_idx].sum()), 1)
    back_to_sig_rate = float(cm[back_idx, sig_idx]) / float(back_total)
    return (
        0.35 * macro_f1
        + 0.30 * float(per_f1[sig_idx])
        + 0.15 * float(per_rec[sig_idx])
        + 0.10 * float(per_prec[sig_idx])
        + 0.10 * float(per_f1[back_idx])
        - 0.25 * sig_to_back_rate
        - 0.05 * back_to_sig_rate
    )


def load_and_prepare_data(
    data_path: str,
) -> tuple[
    np.ndarray,
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:
    """
    Load and prepare data for XGBoost training.

    Args:
        data_path: Path to the parquet file containing page data

    Returns:
        Tuple of
        (features, feature_names, labels, agreement_uuids, announcement_years, orders, split_df)
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
    announcement_years = np.asarray(
        announcement_dates.dt.year.to_numpy(), dtype=np.int32
    )
    agreement_uuids = np.asarray(df["agreement_uuid"].astype(str).to_numpy(), dtype=str)
    order_series = cast(pd.Series, pd.to_numeric(df["order"], errors="raise"))
    orders = np.asarray(order_series.to_numpy(), dtype=np.float32)
    year_counts = cast(
        pd.Series,
        pd.DataFrame(
            {"agreement_uuid": agreement_uuids, "announcement_year": announcement_years}
        )
        .groupby("agreement_uuid")["announcement_year"]
        .nunique()
        .astype(int),
    )
    inconsistent = cast(pd.Series, year_counts[year_counts > 1])
    if not inconsistent.empty:
        raise ValueError(
            "Found agreements spanning multiple announcement years; cannot stratify by year."
        )

    # Map labels to integers
    labels = CLASSIFIER_LABEL_LIST
    label2idx = {label: idx for idx, label in enumerate(labels)}
    label_series = cast(pd.Series, df["label"])
    y_series = label_series.map(cast(Mapping[str, int], label2idx))
    if y_series.isna().any():
        raise ValueError("Found labels missing from CLASSIFIER_LABEL_LIST.")
    y = y_series.astype(int).to_numpy()

    print(f"[data] extracting features in parallel (neighbor_context={USE_NEIGHBOR_CONTEXT})...")
    features, feature_names, agreements_for_features = build_feature_matrix(
        df,
        include_neighbor_context=USE_NEIGHBOR_CONTEXT,
    )
    if not np.array_equal(agreement_uuids, agreements_for_features):
        raise ValueError("Feature extraction changed row order unexpectedly.")
    features = features.astype(np.float32, copy=False)

    split_df = cast(
        pd.DataFrame,
        df[["agreement_uuid", "date_announcement", "label", "order"]].copy(),
    )
    return features, feature_names, y, agreement_uuids, announcement_years, orders, split_df


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
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    hard_negative_mask_train: np.ndarray | None,
    use_hard_negatives: bool,
) -> float:
    """
    Optuna objective function for hyperparameter optimization.

    Args:
        trial: Optuna trial object
        X_train: Training feature matrix
        y_train: Training labels
        X_val: Validation feature matrix
        y_val: Validation labels
        feature_names: Ordered feature name list for DMatrix
        hard_negative_mask_train: Boolean mask for hard negative train rows
        use_hard_negatives: Whether to tune hard-negative sample weighting

    Returns:
        Sig-critical validation score
    """
    # Define hyperparameter search space
    sig_weight_scale = trial.suggest_float("sig_weight_scale", 0.35, 1.4)
    back_weight_scale = trial.suggest_float("back_weight_scale", 0.8, 1.8)
    hard_negative_scale = (
        trial.suggest_float("hard_negative_scale", 1.1, 4.0)
        if use_hard_negatives
        else 1.0
    )
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
    class_w = _class_weight_vector(
        y_train,
        len(CLASSIFIER_LABEL_LIST),
        sig_weight_scale=sig_weight_scale,
        back_weight_scale=back_weight_scale,
    )
    train_weights = class_w[y_train].astype(np.float32, copy=False)
    if use_hard_negatives:
        if hard_negative_mask_train is None:
            raise ValueError("hard_negative_mask_train is required when hard negatives are enabled.")
        if hard_negative_mask_train.shape[0] != y_train.shape[0]:
            raise ValueError("hard_negative_mask_train length must match y_train length.")
        hard_multiplier = np.ones_like(train_weights, dtype=np.float32)
        hard_multiplier[hard_negative_mask_train] = float(hard_negative_scale)
        train_weights = train_weights * hard_multiplier
    dtrain = xgb.DMatrix(
        X_train, label=y_train, weight=train_weights, feature_names=feature_names
    )
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    pruning_callback = XGBoostPruningCallback(trial, "val-f1_macro")

    # Train model
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain, "train"), (dval, "val")],
        custom_metric=macro_f1_eval,
        maximize=True,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
        callbacks=[pruning_callback],
    )

    best_it = getattr(model, "best_iteration", None)
    if best_it is None:
        best_it = getattr(model, "num_boosted_rounds", lambda: 500)()
    trial.set_user_attr("best_iteration", int(best_it))
    trial.set_user_attr("sig_weight_scale", float(sig_weight_scale))
    trial.set_user_attr("back_weight_scale", float(back_weight_scale))
    trial.set_user_attr("hard_negative_scale", float(hard_negative_scale))

    # Evaluate
    y_prob = model.predict(dval)
    y_pred = np.argmax(y_prob, axis=1)
    macro_f1 = float(f1_score(y_val, y_pred, average="macro"))
    score = _sig_critical_score(y_val, y_pred)
    trial.set_user_attr("val_macro_f1", macro_f1)
    trial.set_user_attr("val_sig_critical_score", score)
    return score


@dataclass
class RouterModelArtifact:
    model: xgb.Booster | None
    case_local_to_global: dict[int, int]
    constant_case: int | None
    best_iteration: int
    case_accuracy_val: float | None


def _class_weight_vector_balanced(y_train: np.ndarray, num_classes: int) -> np.ndarray:
    counts_train = np.bincount(y_train, minlength=num_classes).astype(float)
    class_w = counts_train.sum() / np.maximum(counts_train, 1.0)
    class_w = class_w * (num_classes / class_w.sum())
    return class_w


def _split_masks(
    agreement_uuids: np.ndarray,
    split: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_raw = split.get("train")
    val_raw = split.get("val")
    test_raw = split.get("test")
    if not isinstance(train_raw, Sequence) or not isinstance(val_raw, Sequence) or not isinstance(test_raw, Sequence):
        raise ValueError("Split manifest must include sequence keys: train/val/test.")
    train_ids = [str(x) for x in train_raw]
    val_ids = [str(x) for x in val_raw]
    test_ids = [str(x) for x in test_raw]
    train_mask = np.isin(agreement_uuids, train_ids)
    val_mask = np.isin(agreement_uuids, val_ids)
    test_mask = np.isin(agreement_uuids, test_ids)
    return train_mask, val_mask, test_mask


def _parse_fraction_grid(
    raw: str,
    *,
    default: list[float],
    name: str,
) -> list[float]:
    cleaned = raw.strip()
    if not cleaned:
        return list(default)
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if not parts:
        return list(default)
    values: list[float] = []
    for part in parts:
        value = float(part)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} values must be in [0,1], got {value}.")
        values.append(value)
    deduped: list[float] = []
    seen: set[float] = set()
    for value in values:
        key = round(value, 8)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped if deduped else list(default)


def _run_multiclass_optuna(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    num_classes: int,
    trials: int,
    focus_idx: int | None = None,
    score_fn: str = "macro_f1",
) -> tuple[dict[str, float | int], int, float]:
    def _trial_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        macro = float(f1_score(y_true, y_pred, average="macro"))
        if focus_idx is None:
            return macro
        per_prec_arr, per_rec_arr, per_f1_arr, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=np.arange(num_classes),
            average=None,
            zero_division=cast(str, cast(object, 0)),
        )
        per_prec = np.asarray(per_prec_arr)
        per_rec = np.asarray(per_rec_arr)
        per_f1 = np.asarray(per_f1_arr)
        if score_fn == "sig_focus":
            return (
                0.45 * macro
                + 0.35 * float(per_f1[focus_idx])
                + 0.10 * float(per_prec[focus_idx])
                + 0.10 * float(per_rec[focus_idx])
            )
        return macro

    def _objective(trial: optuna.Trial) -> float:
        focus_weight_scale = (
            trial.suggest_float("focus_weight_scale", 0.5, 2.5)
            if focus_idx is not None
            else 1.0
        )
        params: dict[str, float | int | str] = {
            "objective": "multi:softprob",
            "num_class": num_classes,
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
        class_w = _class_weight_vector_balanced(y_train, num_classes)
        if focus_idx is not None:
            class_w[focus_idx] = class_w[focus_idx] * float(focus_weight_scale)
            class_w = class_w * (num_classes / class_w.sum())
        train_weights = class_w[y_train].astype(np.float32, copy=False)
        dtrain = xgb.DMatrix(
            X_train,
            label=y_train,
            weight=train_weights,
            feature_names=feature_names,
        )
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
        best_it = getattr(model, "best_iteration", None)
        if best_it is None:
            best_it = getattr(model, "num_boosted_rounds", lambda: 300)()
        trial.set_user_attr("best_iteration", int(best_it))
        y_prob = model.predict(dval)
        y_pred = np.argmax(y_prob, axis=1)
        score = _trial_score(y_val, y_pred)
        trial.set_user_attr("val_score", float(score))
        trial.set_user_attr("focus_weight_scale", float(focus_weight_scale))
        return score

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
    study.optimize(_objective, n_trials=trials)
    best_params = cast(dict[str, float | int], study.best_params.copy())
    best_it = int(study.best_trial.user_attrs.get("best_iteration", NUM_BOOST_ROUND))
    best_score = float(study.best_trial.user_attrs.get("val_score", study.best_value))
    return best_params, best_it, best_score


def _run_binary_optuna(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    trials: int,
) -> tuple[dict[str, float | int], int, float]:
    def _objective(trial: optuna.Trial) -> float:
        back_weight_scale = trial.suggest_float("back_weight_scale", 0.8, 3.0)
        params: dict[str, float | int | str] = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
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
        class_w = _class_weight_vector_balanced(y_train, 2)
        class_w[1] = class_w[1] * float(back_weight_scale)
        class_w = class_w * (2.0 / class_w.sum())
        train_weights = class_w[y_train].astype(np.float32, copy=False)
        dtrain = xgb.DMatrix(
            X_train,
            label=y_train,
            weight=train_weights,
            feature_names=feature_names,
        )
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
        best_it = getattr(model, "best_iteration", None)
        if best_it is None:
            best_it = getattr(model, "num_boosted_rounds", lambda: 300)()
        trial.set_user_attr("best_iteration", int(best_it))
        p_back = model.predict(dval)
        y_pred = (p_back >= 0.5).astype(int)
        per_prec_arr, per_rec_arr, per_f1_arr, _ = precision_recall_fscore_support(
            y_val,
            y_pred,
            labels=np.arange(2),
            average=None,
            zero_division=cast(str, cast(object, 0)),
        )
        per_prec = np.asarray(per_prec_arr)
        per_rec = np.asarray(per_rec_arr)
        per_f1 = np.asarray(per_f1_arr)
        score = (
            0.45 * float(per_f1[1])
            + 0.35 * float(per_rec[1])
            + 0.20 * float(per_prec[1])
        )
        trial.set_user_attr("val_score", score)
        return score

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
    study.optimize(_objective, n_trials=trials)
    best_params = cast(dict[str, float | int], study.best_params.copy())
    best_it = int(study.best_trial.user_attrs.get("best_iteration", NUM_BOOST_ROUND))
    best_score = float(study.best_trial.user_attrs.get("val_score", study.best_value))
    return best_params, best_it, best_score


def _train_multiclass_model(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_full: np.ndarray,
    y_full: np.ndarray,
    feature_names: list[str],
    params: Mapping[str, float | int],
    best_iteration: int,
    num_classes: int,
    train_path: str,
    prod_path: str,
    focus_idx: int | None = None,
) -> tuple[xgb.Booster, xgb.Booster]:
    xgb_params: dict[str, float | int | str] = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "seed": SEED,
        "nthread": 1,
    }
    focus_weight_scale = float(params.get("focus_weight_scale", 1.0))
    training_params = {
        k: v for k, v in params.items() if k != "focus_weight_scale"
    }
    xgb_params.update(training_params)
    class_w_train = _class_weight_vector_balanced(y_train, num_classes)
    class_w_full = _class_weight_vector_balanced(y_full, num_classes)
    if focus_idx is not None:
        class_w_train[focus_idx] = class_w_train[focus_idx] * focus_weight_scale
        class_w_train = class_w_train * (num_classes / class_w_train.sum())
        class_w_full[focus_idx] = class_w_full[focus_idx] * focus_weight_scale
        class_w_full = class_w_full * (num_classes / class_w_full.sum())
    w_train = class_w_train[y_train].astype(np.float32, copy=False)
    w_full = class_w_full[y_full].astype(np.float32, copy=False)
    dtrain = xgb.DMatrix(
        X_train, label=y_train, weight=w_train, feature_names=feature_names
    )
    dfull = xgb.DMatrix(X_full, label=y_full, weight=w_full, feature_names=feature_names)
    train_model = xgb.train(xgb_params, dtrain, num_boost_round=best_iteration)
    train_model.save_model(train_path)
    full_model = xgb.train(xgb_params, dfull, num_boost_round=best_iteration)
    full_model.save_model(prod_path)
    return train_model, full_model


def _train_binary_model(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_full: np.ndarray,
    y_full: np.ndarray,
    feature_names: list[str],
    params: Mapping[str, float | int],
    best_iteration: int,
    train_path: str,
    prod_path: str,
) -> tuple[xgb.Booster, xgb.Booster]:
    xgb_params: dict[str, float | int | str] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "seed": SEED,
        "nthread": 1,
    }
    back_weight_scale = float(params.get("back_weight_scale", 1.0))
    training_params = {k: v for k, v in params.items() if k != "back_weight_scale"}
    xgb_params.update(training_params)

    class_w_train = _class_weight_vector_balanced(y_train, 2)
    class_w_full = _class_weight_vector_balanced(y_full, 2)
    class_w_train[1] = class_w_train[1] * back_weight_scale
    class_w_train = class_w_train * (2.0 / class_w_train.sum())
    class_w_full[1] = class_w_full[1] * back_weight_scale
    class_w_full = class_w_full * (2.0 / class_w_full.sum())
    w_train = class_w_train[y_train].astype(np.float32, copy=False)
    w_full = class_w_full[y_full].astype(np.float32, copy=False)

    dtrain = xgb.DMatrix(
        X_train, label=y_train, weight=w_train, feature_names=feature_names
    )
    dfull = xgb.DMatrix(X_full, label=y_full, weight=w_full, feature_names=feature_names)
    train_model = xgb.train(xgb_params, dtrain, num_boost_round=best_iteration)
    train_model.save_model(train_path)
    full_model = xgb.train(xgb_params, dfull, num_boost_round=best_iteration)
    full_model.save_model(prod_path)
    return train_model, full_model


def _predict_tail_probs(
    model: xgb.Booster, X: np.ndarray, feature_names: list[str]
) -> np.ndarray:
    dmat = xgb.DMatrix(X, feature_names=feature_names)
    p_back = model.predict(dmat)
    p_back = np.asarray(p_back, dtype=np.float32).reshape(-1)
    p_back = np.clip(p_back, 1e-7, 1.0 - 1e-7)
    p_body = 1.0 - p_back
    return np.column_stack([p_body, p_back]).astype(np.float32, copy=False)


def _doc_groups_for_subset(
    agreement_uuids: np.ndarray,
    orders: np.ndarray,
    subset_indices: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    if subset_indices.ndim != 1:
        raise ValueError("subset_indices must be 1-D.")
    local = np.arange(subset_indices.shape[0], dtype=int)
    temp = pd.DataFrame(
        {
            "agreement_uuid": agreement_uuids[subset_indices].astype(str),
            "order": orders[subset_indices].astype(float),
            "local_idx": local,
        }
    )
    out: list[tuple[str, np.ndarray]] = []
    sorted_temp = temp.sort_values(["agreement_uuid", "order"], kind="mergesort")
    for doc_id, group in sorted_temp.groupby("agreement_uuid", sort=False):
        local_idxs = cast(pd.Series, group["local_idx"]).to_numpy(dtype=np.int32)
        out.append((str(doc_id), local_idxs))
    return out


def _doc_case_id(labels: np.ndarray) -> int:
    sig_idx = CLASSIFIER_LABEL_LIST.index("sig")
    back_idx = CLASSIFIER_LABEL_LIST.index("back_matter")
    has_sig = bool(np.any(labels == sig_idx))
    has_back = bool(np.any(labels == back_idx))
    if has_sig and has_back:
        return 2
    if has_sig and not has_back:
        return 1
    if (not has_sig) and has_back:
        return 3
    return 0


def _router_features(
    base_probs: np.ndarray,
    tail_probs: np.ndarray,
    doc_groups: list[tuple[str, np.ndarray]],
    *,
    case_threshold: float,
) -> tuple[np.ndarray, list[str]]:
    base_front = 0
    base_toc = 1
    base_body = 2
    base_sig = 3
    feats: list[np.ndarray] = []
    for _, locs in doc_groups:
        b = base_probs[locs]
        t = tail_probs[locs]
        n = float(max(int(locs.shape[0]), 1))
        base_argmax = b.argmax(axis=1)
        front_frac = float(np.mean(base_argmax == base_front))
        toc_frac = float(np.mean(base_argmax == base_toc))
        body_frac = float(np.mean(base_argmax == base_body))
        sig_frac = float(np.mean(base_argmax == base_sig))
        sig_prob = b[:, base_sig]
        back_prob = t[:, 1]
        tail_len = int(min(5, locs.shape[0]))
        tail_back_mean = float(np.mean(back_prob[-tail_len:])) if tail_len > 0 else 0.0
        sig_peak_pos = float(np.argmax(sig_prob)) / float(max(locs.shape[0] - 1, 1))
        back_peak_pos = float(np.argmax(back_prob)) / float(max(locs.shape[0] - 1, 1))
        feature_vec = np.array(
            [
                n,
                float(np.mean(b[:, base_front])),
                float(np.mean(b[:, base_toc])),
                float(np.mean(b[:, base_body])),
                float(np.mean(sig_prob)),
                float(np.max(sig_prob)),
                float(b[-1, base_sig]),
                float(np.mean(t[:, 0])),
                float(np.mean(back_prob)),
                float(np.max(back_prob)),
                float(back_prob[-1]),
                tail_back_mean,
                float(np.mean(back_prob >= case_threshold)),
                front_frac,
                toc_frac,
                body_frac,
                sig_frac,
                sig_peak_pos,
                back_peak_pos,
            ],
            dtype=np.float32,
        )
        feats.append(feature_vec)
    if not feats:
        raise ValueError("No documents available for router features.")
    names = [
        "doc_len",
        "mean_front",
        "mean_toc",
        "mean_body",
        "mean_sig",
        "max_sig",
        "last_sig",
        "mean_tail_body",
        "mean_tail_back",
        "max_tail_back",
        "last_tail_back",
        "tail_back_last5",
        "tail_back_over_threshold_frac",
        "front_argmax_frac",
        "toc_argmax_frac",
        "body_argmax_frac",
        "sig_argmax_frac",
        "sig_peak_pos",
        "back_peak_pos",
    ]
    return np.vstack(feats).astype(np.float32, copy=False), names


def _router_targets_from_groups(
    y_subset: np.ndarray, doc_groups: list[tuple[str, np.ndarray]]
) -> np.ndarray:
    case_targets: list[int] = []
    for _, locs in doc_groups:
        case_targets.append(_doc_case_id(y_subset[locs]))
    return np.asarray(case_targets, dtype=np.int32)


def _run_router_optuna(
    *,
    X_train_docs: np.ndarray,
    y_train_cases: np.ndarray,
    X_val_docs: np.ndarray,
    y_val_cases: np.ndarray,
    base_val_probs: np.ndarray,
    tail_val_probs: np.ndarray,
    val_doc_groups: list[tuple[str, np.ndarray]],
    y_val_pages_true: np.ndarray,
    blend_candidates: Sequence[float],
    feature_names: list[str],
    trials: int,
) -> tuple[dict[str, float | int], int, float | None, float | None]:
    if len(blend_candidates) == 0:
        raise ValueError("blend_candidates must not be empty.")
    observed_cases = sorted(int(x) for x in np.unique(y_train_cases))
    if trials <= 0 or len(observed_cases) <= 1:
        return {}, MOE_ROUTER_NUM_BOOST_ROUND, None, None

    case_to_local = {case_id: i for i, case_id in enumerate(observed_cases)}
    local_to_case = {i: case_id for case_id, i in case_to_local.items()}
    y_train_local = np.asarray([case_to_local[int(y)] for y in y_train_cases], dtype=np.int32)
    y_val_local = np.asarray([case_to_local.get(int(y), -1) for y in y_val_cases], dtype=np.int32)
    val_valid_mask = y_val_local >= 0
    if not np.any(val_valid_mask):
        return {}, MOE_ROUTER_NUM_BOOST_ROUND, None, None

    X_val_valid = np.asarray(X_val_docs[val_valid_mask], dtype=np.float32)
    y_val_valid = np.asarray(y_val_local[val_valid_mask], dtype=np.int32)
    num_classes = len(observed_cases)

    def _objective(trial: optuna.Trial) -> float:
        params: dict[str, float | int | str] = {
            "objective": "multi:softprob",
            "num_class": num_classes,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "seed": SEED,
            "nthread": 1,
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "eta": trial.suggest_float("eta", 5e-3, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
            "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
            "max_bin": trial.suggest_categorical("max_bin", [128, 256, 384]),
        }
        class_w = _class_weight_vector_balanced(y_train_local, num_classes)
        train_weights = class_w[y_train_local].astype(np.float32, copy=False)
        dtrain = xgb.DMatrix(
            X_train_docs,
            label=y_train_local,
            weight=train_weights,
            feature_names=feature_names,
        )
        dval = xgb.DMatrix(X_val_valid, label=y_val_valid, feature_names=feature_names)
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=MOE_ROUTER_NUM_BOOST_ROUND,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=MOE_ROUTER_EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
        best_it_raw = getattr(model, "best_iteration", None)
        best_rounds = (
            int(getattr(model, "num_boosted_rounds", lambda: MOE_ROUTER_NUM_BOOST_ROUND)())
            if best_it_raw is None
            else int(best_it_raw) + 1
        )
        trial.set_user_attr("best_num_boost_round", int(best_rounds))
        y_val_prob_valid = model.predict(dval)
        y_val_hat_valid = np.argmax(y_val_prob_valid, axis=1)
        router_case_acc = float(np.mean(y_val_hat_valid == y_val_valid))

        dval_all = xgb.DMatrix(X_val_docs, feature_names=feature_names)
        y_val_prob_all = model.predict(dval_all)
        y_val_hat_local_all = np.argmax(y_val_prob_all, axis=1)
        y_val_hat_global_all = np.asarray(
            [local_to_case[int(v)] for v in y_val_hat_local_all],
            dtype=np.int32,
        )
        best_sig_score = float("-inf")
        best_macro_f1 = float("-inf")
        best_blend = float(blend_candidates[0])
        for blend in blend_candidates:
            y_val_pages_pred = _decode_subset_from_probs(
                base_probs=base_val_probs,
                tail_probs=tail_val_probs,
                doc_groups=val_doc_groups,
                case_hat=y_val_hat_global_all,
                body_blend=float(blend),
            )
            sig_score = float(_sig_critical_score(y_val_pages_true, y_val_pages_pred))
            macro_f1 = float(f1_score(y_val_pages_true, y_val_pages_pred, average="macro"))
            better = (sig_score > best_sig_score) or (
                sig_score == best_sig_score and macro_f1 > best_macro_f1
            )
            if better:
                best_sig_score = sig_score
                best_macro_f1 = macro_f1
                best_blend = float(blend)

        trial.set_user_attr("val_router_case_acc", router_case_acc)
        trial.set_user_attr("val_page_macro_f1", best_macro_f1)
        trial.set_user_attr("val_page_sig_critical_score", best_sig_score)
        trial.set_user_attr("best_blend", best_blend)
        return best_sig_score

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
    study.optimize(_objective, n_trials=trials)
    best_params = cast(dict[str, float | int], study.best_params.copy())
    best_rounds = int(
        study.best_trial.user_attrs.get(
            "best_num_boost_round",
            MOE_ROUTER_NUM_BOOST_ROUND,
        )
    )
    best_score = float(
        study.best_trial.user_attrs.get(
            "val_page_sig_critical_score", study.best_value
        )
    )
    best_blend = study.best_trial.user_attrs.get("best_blend")
    best_blend_value = float(best_blend) if best_blend is not None else None
    return best_params, best_rounds, best_score, best_blend_value


def _train_router(
    *,
    X_train_docs: np.ndarray,
    y_train_cases: np.ndarray,
    X_val_docs: np.ndarray,
    y_val_cases: np.ndarray,
    feature_names: list[str],
    train_path: str | None = None,
    params: Mapping[str, float | int] | None = None,
    num_boost_round: int | None = None,
) -> RouterModelArtifact:
    observed_cases = sorted(int(x) for x in np.unique(y_train_cases))
    if not observed_cases:
        raise ValueError("Router training has no observed document cases.")
    if len(observed_cases) == 1:
        constant_case = observed_cases[0]
        return RouterModelArtifact(
            model=None,
            case_local_to_global={0: constant_case},
            constant_case=constant_case,
            best_iteration=0,
            case_accuracy_val=float(np.mean(y_val_cases == constant_case))
            if y_val_cases.size > 0
            else None,
        )

    case_to_local = {case_id: i for i, case_id in enumerate(observed_cases)}
    local_to_case = {i: case_id for case_id, i in case_to_local.items()}
    y_train_local = np.asarray([case_to_local[int(y)] for y in y_train_cases], dtype=np.int32)
    y_val_local = np.asarray([case_to_local.get(int(y), -1) for y in y_val_cases], dtype=np.int32)

    router_params: dict[str, float | int | str] = {
        "objective": "multi:softprob",
        "num_class": len(observed_cases),
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "seed": SEED,
        "nthread": 1,
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1,
        "lambda": 1.0,
        "alpha": 0.0,
        "max_bin": 256,
    }
    if params:
        router_params.update(params)
    rounds = (
        max(int(num_boost_round), 1)
        if num_boost_round is not None
        else MOE_ROUTER_NUM_BOOST_ROUND
    )
    dtrain = xgb.DMatrix(X_train_docs, label=y_train_local, feature_names=feature_names)
    val_valid_mask = y_val_local >= 0
    use_val = bool(y_val_local.size > 0 and np.any(val_valid_mask))
    evals: list[tuple[xgb.DMatrix, str]] = [(dtrain, "train")]
    dval: xgb.DMatrix | None = None
    if use_val:
        dval = xgb.DMatrix(
            X_val_docs[val_valid_mask],
            label=y_val_local[val_valid_mask],
            feature_names=feature_names,
        )
        evals.append((dval, "val"))
    train_model = xgb.train(
        router_params,
        dtrain,
        num_boost_round=rounds,
        evals=evals,
        early_stopping_rounds=MOE_ROUTER_EARLY_STOPPING_ROUNDS if use_val else None,
        verbose_eval=False,
    )
    best_it_raw = getattr(train_model, "best_iteration", None)
    if best_it_raw is None:
        best_it = int(getattr(train_model, "num_boosted_rounds", lambda: rounds)())
    else:
        best_it = int(best_it_raw) + 1
    if train_path:
        train_model.save_model(train_path)

    val_acc: float | None = None
    if dval is not None and use_val:
        y_val_prob = train_model.predict(dval)
        y_val_hat_local = np.argmax(y_val_prob, axis=1)
        val_acc = float(np.mean(y_val_hat_local == y_val_local[val_valid_mask]))

    return RouterModelArtifact(
        model=train_model,
        case_local_to_global=local_to_case,
        constant_case=None,
        best_iteration=best_it,
        case_accuracy_val=val_acc,
    )


def _train_router_full(
    *,
    X_docs: np.ndarray,
    y_cases: np.ndarray,
    feature_names: list[str],
    num_boost_round: int,
    model_path: str | None = None,
    params: Mapping[str, float | int] | None = None,
) -> RouterModelArtifact:
    observed_cases = sorted(int(x) for x in np.unique(y_cases))
    if not observed_cases:
        raise ValueError("Router full training has no observed document cases.")
    if len(observed_cases) == 1:
        constant_case = observed_cases[0]
        return RouterModelArtifact(
            model=None,
            case_local_to_global={0: constant_case},
            constant_case=constant_case,
            best_iteration=0,
            case_accuracy_val=None,
        )
    case_to_local = {case_id: i for i, case_id in enumerate(observed_cases)}
    local_to_case = {i: case_id for case_id, i in case_to_local.items()}
    y_local = np.asarray([case_to_local[int(y)] for y in y_cases], dtype=np.int32)
    router_params: dict[str, float | int | str] = {
        "objective": "multi:softprob",
        "num_class": len(observed_cases),
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "seed": SEED,
        "nthread": 1,
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1,
        "lambda": 1.0,
        "alpha": 0.0,
        "max_bin": 256,
    }
    if params:
        router_params.update(params)
    dtrain = xgb.DMatrix(X_docs, label=y_local, feature_names=feature_names)
    model = xgb.train(
        router_params,
        dtrain,
        num_boost_round=max(int(num_boost_round), 1),
        verbose_eval=False,
    )
    if model_path:
        model.save_model(model_path)
    return RouterModelArtifact(
        model=model,
        case_local_to_global=local_to_case,
        constant_case=None,
        best_iteration=max(int(num_boost_round), 1),
        case_accuracy_val=None,
    )


def _predict_router_cases(
    router: RouterModelArtifact,
    X_docs: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    if router.constant_case is not None:
        return np.full((X_docs.shape[0],), int(router.constant_case), dtype=np.int32)
    if router.model is None:
        raise ValueError("Router model is missing.")
    ddocs = xgb.DMatrix(X_docs, feature_names=feature_names)
    probs = router.model.predict(ddocs)
    local_hat = np.argmax(probs, axis=1)
    return np.asarray(
        [router.case_local_to_global[int(idx)] for idx in local_hat], dtype=np.int32
    )


def _monotone_decode(
    log_probs: np.ndarray,
    states: Sequence[int],
    start_states: Sequence[int],
) -> np.ndarray:
    t_steps = log_probs.shape[0]
    k = len(states)
    if t_steps == 0:
        return np.asarray([], dtype=np.int32)
    state_arr = np.asarray(states, dtype=np.int32)
    start_set = set(int(s) for s in start_states)
    dp = np.full((t_steps, k), -1e12, dtype=np.float64)
    backptr = np.full((t_steps, k), -1, dtype=np.int32)
    for i, s in enumerate(state_arr):
        if int(s) in start_set:
            dp[0, i] = float(log_probs[0, int(s)])
    if np.all(dp[0] <= -1e11):
        for i, s in enumerate(state_arr):
            dp[0, i] = float(log_probs[0, int(s)])
    for t in range(1, t_steps):
        for i, s in enumerate(state_arr):
            prev_slice = dp[t - 1, : i + 1]
            best_prev = int(np.argmax(prev_slice))
            dp[t, i] = float(prev_slice[best_prev] + log_probs[t, int(s)])
            backptr[t, i] = best_prev
    path = np.zeros((t_steps,), dtype=np.int32)
    cur = int(np.argmax(dp[-1]))
    path[-1] = int(state_arr[cur])
    for t in range(t_steps - 1, 0, -1):
        cur = int(backptr[t, cur])
        path[t - 1] = int(state_arr[cur])
    return path


def _monotone_decode_with_required_states(
    log_probs: np.ndarray,
    states: Sequence[int],
    start_states: Sequence[int],
    required_states: Sequence[int],
) -> np.ndarray:
    t_steps = log_probs.shape[0]
    if t_steps == 0:
        return np.asarray([], dtype=np.int32)
    state_arr = np.asarray(states, dtype=np.int32)
    required_state_set = {int(s) for s in required_states}
    req_in_path = [int(s) for s in state_arr if int(s) in required_state_set]
    if not req_in_path:
        return _monotone_decode(log_probs, states, start_states)
    req_to_bit = {state: bit for bit, state in enumerate(sorted(set(req_in_path)))}
    req_mask_full = (1 << len(req_to_bit)) - 1
    k = state_arr.shape[0]
    mask_size = 1 << len(req_to_bit)
    start_set = {int(s) for s in start_states}

    dp = np.full((t_steps, k, mask_size), -1e12, dtype=np.float64)
    prev_state = np.full((t_steps, k, mask_size), -1, dtype=np.int32)
    prev_mask = np.full((t_steps, k, mask_size), -1, dtype=np.int32)

    for i, state in enumerate(state_arr):
        state_int = int(state)
        if state_int not in start_set:
            continue
        emit = float(log_probs[0, state_int])
        init_mask = 0
        if state_int in req_to_bit:
            init_mask |= 1 << req_to_bit[state_int]
        dp[0, i, init_mask] = emit

    if np.all(dp[0] <= -1e11):
        for i, state in enumerate(state_arr):
            state_int = int(state)
            emit = float(log_probs[0, state_int])
            init_mask = 0
            if state_int in req_to_bit:
                init_mask |= 1 << req_to_bit[state_int]
            dp[0, i, init_mask] = emit

    for t in range(1, t_steps):
        for i, state in enumerate(state_arr):
            state_int = int(state)
            emit = float(log_probs[t, state_int])
            add_bit = 0
            if state_int in req_to_bit:
                add_bit = 1 << req_to_bit[state_int]
            for mask_prev in range(mask_size):
                prev_scores = dp[t - 1, : i + 1, mask_prev]
                best_prev_idx = int(np.argmax(prev_scores))
                best_prev_score = float(prev_scores[best_prev_idx])
                if best_prev_score <= -1e11:
                    continue
                mask_new = mask_prev | add_bit
                cand = best_prev_score + emit
                if cand > dp[t, i, mask_new]:
                    dp[t, i, mask_new] = cand
                    prev_state[t, i, mask_new] = best_prev_idx
                    prev_mask[t, i, mask_new] = mask_prev

    end_scores = dp[t_steps - 1, :, req_mask_full]
    if np.all(end_scores <= -1e11):
        return _monotone_decode(log_probs, states, start_states)
    end_i = int(np.argmax(end_scores))
    end_mask = req_mask_full

    path = np.zeros((t_steps,), dtype=np.int32)
    path[t_steps - 1] = int(state_arr[end_i])
    cur_i = end_i
    cur_mask = end_mask
    for t in range(t_steps - 1, 0, -1):
        prev_i = int(prev_state[t, cur_i, cur_mask])
        prev_m = int(prev_mask[t, cur_i, cur_mask])
        if prev_i < 0 or prev_m < 0:
            return _monotone_decode(log_probs, states, start_states)
        path[t - 1] = int(state_arr[prev_i])
        cur_i = prev_i
        cur_mask = prev_m
    return path


def _decode_doc_moe(
    base_doc_probs: np.ndarray,
    tail_doc_probs: np.ndarray,
    case_id: int,
    *,
    body_blend: float,
) -> np.ndarray:
    front_idx = CLASSIFIER_LABEL_LIST.index("front_matter")
    toc_idx = CLASSIFIER_LABEL_LIST.index("toc")
    body_idx = CLASSIFIER_LABEL_LIST.index("body")
    sig_idx = CLASSIFIER_LABEL_LIST.index("sig")
    back_idx = CLASSIFIER_LABEL_LIST.index("back_matter")
    probs = np.zeros((base_doc_probs.shape[0], len(CLASSIFIER_LABEL_LIST)), dtype=np.float32)
    probs[:, front_idx] = base_doc_probs[:, 0]
    probs[:, toc_idx] = base_doc_probs[:, 1]
    probs[:, sig_idx] = base_doc_probs[:, 3]
    body_mix = body_blend * base_doc_probs[:, 2] + (1.0 - body_blend) * tail_doc_probs[:, 0]
    probs[:, body_idx] = body_mix
    probs[:, back_idx] = tail_doc_probs[:, 1]
    probs = np.clip(probs, 1e-9, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    log_probs = np.log(probs)

    if case_id == 0:  # no sig, no back
        states = [front_idx, toc_idx, body_idx]
        required_states: list[int] = []
    elif case_id == 1:  # sig, no back
        states = [front_idx, toc_idx, body_idx, sig_idx]
        required_states = [sig_idx]
    elif case_id == 2:  # sig, back
        states = [front_idx, toc_idx, body_idx, sig_idx, back_idx]
        required_states = [sig_idx, back_idx]
    elif case_id == 3:  # no sig, back
        states = [front_idx, toc_idx, body_idx, back_idx]
        required_states = [back_idx]
    else:
        raise ValueError(f"Unexpected router case id: {case_id}")
    start_states = [front_idx, toc_idx, body_idx]
    return _monotone_decode_with_required_states(
        log_probs,
        states,
        start_states,
        required_states,
    )


def _page_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, object]:
    num_classes = len(CLASSIFIER_LABEL_LIST)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=cast(str, cast(object, 0)),
    )
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    class_acc = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    per_prec_arr, per_rec_arr, per_f1_arr, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
        average=None,
        zero_division=cast(str, cast(object, 0)),
    )
    per_prec = np.asarray(per_prec_arr)
    per_rec = np.asarray(per_rec_arr)
    per_f1 = np.asarray(per_f1_arr)
    return {
        "overall": {
            "accuracy": float(accuracy),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
        },
        "confusion_matrix": cm.astype(int).tolist(),
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


def _print_metrics(tag: str, metrics: Mapping[str, object]) -> None:
    overall = cast(Mapping[str, object], metrics["overall"])
    overall_acc = float(cast(float, overall["accuracy"]))
    overall_prec = float(cast(float, overall["precision_macro"]))
    overall_rec = float(cast(float, overall["recall_macro"]))
    overall_f1 = float(cast(float, overall["f1_macro"]))
    print(f"{tag}  Acc: {overall_acc:.4f}  P/R/F1: {overall_prec:.4f}/{overall_rec:.4f}/{overall_f1:.4f}")
    print(f"{tag} Confusion Matrix:")
    print(np.asarray(metrics["confusion_matrix"]))
    print(f"{tag} Per-class metrics:")
    per_class = cast(Mapping[str, Mapping[str, object]], metrics["per_class"])
    for label in CLASSIFIER_LABEL_LIST:
        values = per_class[label]
        class_acc = float(cast(float, values["accuracy"]))
        class_prec = float(cast(float, values["precision"]))
        class_rec = float(cast(float, values["recall"]))
        class_f1 = float(cast(float, values["f1"]))
        print(f"  {label}: Acc={class_acc:.4f} P={class_prec:.4f} R={class_rec:.4f} F1={class_f1:.4f}")


def _decode_subset_from_probs(
    *,
    base_probs: np.ndarray,
    tail_probs: np.ndarray,
    doc_groups: list[tuple[str, np.ndarray]],
    case_hat: np.ndarray,
    body_blend: float,
) -> np.ndarray:
    y_hat_subset = np.zeros((base_probs.shape[0],), dtype=np.int32)
    for doc_i, (_, locs) in enumerate(doc_groups):
        case_id = int(case_hat[doc_i])
        decoded = _decode_doc_moe(
            base_probs[locs],
            tail_probs[locs],
            case_id,
            body_blend=body_blend,
        )
        y_hat_subset[locs] = decoded
    return y_hat_subset


def _predict_moe_subset(
    *,
    base_model: xgb.Booster,
    tail_model: xgb.Booster,
    router: RouterModelArtifact,
    body_blend: float,
    case_threshold: float,
    feature_names: list[str],
    features: np.ndarray,
    labels: np.ndarray,
    agreement_uuids: np.ndarray,
    orders: np.ndarray,
    subset_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    subset_idx = np.where(subset_mask)[0]
    X_subset = np.asarray(features[subset_idx])
    y_subset = np.asarray(labels[subset_idx])
    base_probs = base_model.predict(xgb.DMatrix(X_subset, feature_names=feature_names))
    tail_probs = _predict_tail_probs(tail_model, X_subset, feature_names)
    doc_groups = _doc_groups_for_subset(agreement_uuids, orders, subset_idx)
    X_docs, router_feature_names = _router_features(
        base_probs,
        tail_probs,
        doc_groups,
        case_threshold=case_threshold,
    )
    case_hat = _predict_router_cases(router, X_docs, router_feature_names)
    case_true = _router_targets_from_groups(y_subset, doc_groups)
    y_hat_subset = _decode_subset_from_probs(
        base_probs=base_probs,
        tail_probs=tail_probs,
        doc_groups=doc_groups,
        case_hat=case_hat,
        body_blend=body_blend,
    )
    return y_subset, y_hat_subset, case_hat, case_true


def _run_moe_training(
    *,
    data_path: str,
    split_path: str,
    year_window: int,
    length_bucket_edges: list[float],
    back_matter_bucket_edges: list[float],
) -> None:
    blend_candidates = _parse_fraction_grid(
        MOE_BLEND_GRID_RAW,
        default=[MOE_BODY_BLEND],
        name="MOE blend grid",
    )
    case_threshold_candidates = _parse_fraction_grid(
        MOE_CASE_THRESHOLD_GRID_RAW,
        default=[MOE_CASE_THRESHOLD],
        name="MOE case-threshold grid",
    )
    if MOE_BODY_BLEND not in blend_candidates:
        blend_candidates.append(MOE_BODY_BLEND)
    if MOE_CASE_THRESHOLD not in case_threshold_candidates:
        case_threshold_candidates.append(MOE_CASE_THRESHOLD)
    if not MOE_TUNE_ON_VAL:
        blend_candidates = [MOE_BODY_BLEND]
        case_threshold_candidates = [MOE_CASE_THRESHOLD]

    print(
        f"[moe] training mode enabled (base_trials={MOE_BASE_TRIALS}, tail_trials={MOE_TAIL_TRIALS}, tune_on_val={MOE_TUNE_ON_VAL})"
    )
    (
        features,
        feature_names,
        y,
        agreement_uuids,
        _years,
        orders,
        split_df,
    ) = load_and_prepare_data(data_path)
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

    if "train" not in split or "val" not in split or "test" not in split:
        raise ValueError("Split manifest missing required keys: train/val/test.")
    train_mask, val_mask, test_mask = _split_masks(agreement_uuids, split)
    train_or_val_mask = train_mask | val_mask

    front_idx = CLASSIFIER_LABEL_LIST.index("front_matter")
    toc_idx = CLASSIFIER_LABEL_LIST.index("toc")
    body_idx = CLASSIFIER_LABEL_LIST.index("body")
    sig_idx = CLASSIFIER_LABEL_LIST.index("sig")
    back_idx = CLASSIFIER_LABEL_LIST.index("back_matter")

    base_global_to_local = {
        front_idx: 0,
        toc_idx: 1,
        body_idx: 2,
        sig_idx: 3,
    }
    base_local_to_global = {v: k for k, v in base_global_to_local.items()}
    base_train_rows = train_mask & (y != back_idx)
    base_val_rows = val_mask & (y != back_idx)
    base_full_rows = train_or_val_mask & (y != back_idx)
    if not np.any(base_train_rows) or not np.any(base_val_rows):
        raise ValueError("Base expert train/val rows are empty.")
    y_base_train = np.asarray([base_global_to_local[int(label)] for label in y[base_train_rows]])
    y_base_val = np.asarray([base_global_to_local[int(label)] for label in y[base_val_rows]])
    y_base_full = np.asarray([base_global_to_local[int(label)] for label in y[base_full_rows]])

    tail_rows_all = np.isin(y, [body_idx, back_idx])
    tail_train_rows = train_mask & tail_rows_all
    tail_val_rows = val_mask & tail_rows_all
    tail_full_rows = train_or_val_mask & tail_rows_all
    if not np.any(tail_train_rows) or not np.any(tail_val_rows):
        raise ValueError("Tail expert train/val rows are empty.")
    y_tail_train = (y[tail_train_rows] == back_idx).astype(np.int32)
    y_tail_val = (y[tail_val_rows] == back_idx).astype(np.int32)
    y_tail_full = (y[tail_full_rows] == back_idx).astype(np.int32)

    print("[moe] optimizing base expert...")
    base_best_params, base_best_it, base_best_score = _run_multiclass_optuna(
        X_train=np.asarray(features[base_train_rows]),
        y_train=y_base_train,
        X_val=np.asarray(features[base_val_rows]),
        y_val=y_base_val,
        feature_names=feature_names,
        num_classes=4,
        trials=MOE_BASE_TRIALS,
        focus_idx=3,
        score_fn="sig_focus",
    )
    print(f"[moe] base best score={base_best_score:.5f} best_iter={base_best_it}")
    base_train_model, base_full_model = _train_multiclass_model(
        X_train=np.asarray(features[base_train_rows]),
        y_train=y_base_train,
        X_full=np.asarray(features[base_full_rows]),
        y_full=y_base_full,
        feature_names=feature_names,
        params=base_best_params,
        best_iteration=base_best_it,
        num_classes=4,
        train_path=CLASSIFIER_XGB_MOE_BASE_TRAIN_PATH,
        prod_path=CLASSIFIER_XGB_MOE_BASE_PATH,
        focus_idx=3,
    )

    print("[moe] optimizing tail expert (body/back)...")
    tail_best_params, tail_best_it, tail_best_score = _run_binary_optuna(
        X_train=np.asarray(features[tail_train_rows]),
        y_train=y_tail_train,
        X_val=np.asarray(features[tail_val_rows]),
        y_val=y_tail_val,
        feature_names=feature_names,
        trials=MOE_TAIL_TRIALS,
    )
    print(f"[moe] tail best score={tail_best_score:.5f} best_iter={tail_best_it}")
    tail_train_model, tail_full_model = _train_binary_model(
        X_train=np.asarray(features[tail_train_rows]),
        y_train=y_tail_train,
        X_full=np.asarray(features[tail_full_rows]),
        y_full=y_tail_full,
        feature_names=feature_names,
        params=tail_best_params,
        best_iteration=tail_best_it,
        train_path=CLASSIFIER_XGB_MOE_TAIL_TRAIN_PATH,
        prod_path=CLASSIFIER_XGB_MOE_TAIL_PATH,
    )

    print(f"[moe] tuning router/decode on val (case_thresholds={case_threshold_candidates}, blends={blend_candidates})...")
    train_subset_idx = np.where(train_mask)[0]
    val_subset_idx = np.where(val_mask)[0]
    X_train_subset = np.asarray(features[train_subset_idx])
    X_val_subset = np.asarray(features[val_subset_idx])
    base_train_probs = base_train_model.predict(
        xgb.DMatrix(X_train_subset, feature_names=feature_names)
    )
    base_val_probs = base_train_model.predict(
        xgb.DMatrix(X_val_subset, feature_names=feature_names)
    )
    tail_train_probs = _predict_tail_probs(tail_train_model, X_train_subset, feature_names)
    tail_val_probs = _predict_tail_probs(tail_train_model, X_val_subset, feature_names)
    train_doc_groups = _doc_groups_for_subset(agreement_uuids, orders, train_subset_idx)
    val_doc_groups = _doc_groups_for_subset(agreement_uuids, orders, val_subset_idx)
    y_train_cases = _router_targets_from_groups(y[train_subset_idx], train_doc_groups)
    y_val_cases = _router_targets_from_groups(y[val_subset_idx], val_doc_groups)
    y_val_pages_true = np.asarray(y[val_subset_idx])
    best_score = float("-inf")
    best_macro_f1 = float("-inf")
    selected_body_blend = float(MOE_BODY_BLEND)
    selected_case_threshold = float(MOE_CASE_THRESHOLD)
    selected_router_train_artifact: RouterModelArtifact | None = None
    selected_router_params: dict[str, float | int] = {}
    selected_router_optuna_score: float | None = None
    selected_router_optuna_best_blend: float | None = None
    calibration_results: list[dict[str, float]] = []

    for case_threshold in case_threshold_candidates:
        X_train_docs, router_feature_names = _router_features(
            base_train_probs,
            tail_train_probs,
            train_doc_groups,
            case_threshold=case_threshold,
        )
        X_val_docs, _ = _router_features(
            base_val_probs,
            tail_val_probs,
            val_doc_groups,
            case_threshold=case_threshold,
        )
        (
            router_best_params,
            router_best_rounds,
            router_optuna_score,
            router_optuna_best_blend,
        ) = _run_router_optuna(
            X_train_docs=X_train_docs,
            y_train_cases=y_train_cases,
            X_val_docs=X_val_docs,
            y_val_cases=y_val_cases,
            base_val_probs=base_val_probs,
            tail_val_probs=tail_val_probs,
            val_doc_groups=val_doc_groups,
            y_val_pages_true=y_val_pages_true,
            blend_candidates=blend_candidates,
            feature_names=router_feature_names,
            trials=MOE_ROUTER_TRIALS,
        )
        router_train_artifact = _train_router(
            X_train_docs=X_train_docs,
            y_train_cases=y_train_cases,
            X_val_docs=X_val_docs,
            y_val_cases=y_val_cases,
            feature_names=router_feature_names,
            train_path=None,
            params=router_best_params,
            num_boost_round=router_best_rounds,
        )
        val_case_hat = _predict_router_cases(
            router_train_artifact,
            X_val_docs,
            router_feature_names,
        )
        router_case_acc = float(np.mean(val_case_hat == y_val_cases))
        for body_blend in blend_candidates:
            y_val_pred = _decode_subset_from_probs(
                base_probs=base_val_probs,
                tail_probs=tail_val_probs,
                doc_groups=val_doc_groups,
                case_hat=val_case_hat,
                body_blend=body_blend,
            )
            score = float(_sig_critical_score(y_val_pages_true, y_val_pred))
            macro_f1 = float(f1_score(y_val_pages_true, y_val_pred, average="macro"))
            calibration_results.append(
                {
                    "case_threshold": float(case_threshold),
                    "body_blend": float(body_blend),
                    "sig_critical_score": score,
                    "macro_f1": macro_f1,
                    "router_case_accuracy": router_case_acc,
                    "router_optuna_score": (
                        float(router_optuna_score)
                        if router_optuna_score is not None
                        else -1.0
                    ),
                    "router_optuna_best_blend": (
                        float(router_optuna_best_blend)
                        if router_optuna_best_blend is not None
                        else -1.0
                    ),
                    "router_best_iteration": float(router_train_artifact.best_iteration),
                }
            )
            better = (score > best_score) or (
                score == best_score and macro_f1 > best_macro_f1
            )
            if better:
                best_score = score
                best_macro_f1 = macro_f1
                selected_body_blend = float(body_blend)
                selected_case_threshold = float(case_threshold)
                selected_router_train_artifact = router_train_artifact
                selected_router_params = dict(router_best_params)
                selected_router_optuna_score = router_optuna_score
                selected_router_optuna_best_blend = router_optuna_best_blend

    if selected_router_train_artifact is None:
        raise RuntimeError("Failed to select MoE calibration hyperparameters.")

    if selected_router_train_artifact.model is not None:
        selected_router_train_artifact.model.save_model(
            CLASSIFIER_XGB_MOE_ROUTER_TRAIN_PATH
        )
    print(f"[moe] selected calibration case_threshold={selected_case_threshold:.3f} body_blend={selected_body_blend:.3f} val_sig_critical_score={best_score:.5f} val_macro_f1={best_macro_f1:.5f}")

    full_subset_idx = np.where(train_or_val_mask)[0]
    X_full_subset = np.asarray(features[full_subset_idx])
    base_full_probs = base_full_model.predict(
        xgb.DMatrix(X_full_subset, feature_names=feature_names)
    )
    tail_full_probs = _predict_tail_probs(tail_full_model, X_full_subset, feature_names)
    full_doc_groups = _doc_groups_for_subset(agreement_uuids, orders, full_subset_idx)
    X_full_docs, router_feature_names_full = _router_features(
        base_full_probs,
        tail_full_probs,
        full_doc_groups,
        case_threshold=selected_case_threshold,
    )
    y_full_cases = _router_targets_from_groups(y[full_subset_idx], full_doc_groups)
    router_full_artifact = _train_router_full(
        X_docs=X_full_docs,
        y_cases=y_full_cases,
        feature_names=router_feature_names_full,
        num_boost_round=max(selected_router_train_artifact.best_iteration, 1),
        model_path=CLASSIFIER_XGB_MOE_ROUTER_PATH,
        params=selected_router_params,
    )

    print("[moe] evaluating validation split with train-only experts...")
    y_val_true, y_val_pred, y_val_case_hat, y_val_case_true = _predict_moe_subset(
        base_model=base_train_model,
        tail_model=tail_train_model,
        router=selected_router_train_artifact,
        body_blend=selected_body_blend,
        case_threshold=selected_case_threshold,
        feature_names=feature_names,
        features=features,
        labels=y,
        agreement_uuids=agreement_uuids,
        orders=orders,
        subset_mask=val_mask,
    )
    val_metrics = _page_metrics(y_val_true, y_val_pred)
    _print_metrics("Val (moe)", val_metrics)

    print("[moe] evaluating test split with production experts...")
    y_test_true, y_test_pred, y_test_case_hat, y_test_case_true = _predict_moe_subset(
        base_model=base_full_model,
        tail_model=tail_full_model,
        router=router_full_artifact,
        body_blend=selected_body_blend,
        case_threshold=selected_case_threshold,
        feature_names=feature_names,
        features=features,
        labels=y,
        agreement_uuids=agreement_uuids,
        orders=orders,
        subset_mask=test_mask,
    )
    test_metrics = _page_metrics(y_test_true, y_test_pred)
    _print_metrics("Test (moe)", test_metrics)

    case_accuracy_val = (
        float(selected_router_train_artifact.case_accuracy_val)
        if selected_router_train_artifact.case_accuracy_val is not None
        else None
    )
    case_accuracy_val_runtime = float(np.mean(y_val_case_hat == y_val_case_true))
    case_accuracy_test = float(np.mean(y_test_case_hat == y_test_case_true))
    metrics = {
        "mode": "xgb_moe_v1",
        "split": {
            "path": split_path,
            "year_window": int(year_window),
        },
        "optimization": {
            "tune_on_val": bool(MOE_TUNE_ON_VAL),
            "blend_candidates": [float(x) for x in blend_candidates],
            "case_threshold_candidates": [float(x) for x in case_threshold_candidates],
            "base_trials": int(MOE_BASE_TRIALS),
            "tail_trials": int(MOE_TAIL_TRIALS),
            "selected_body_blend": float(selected_body_blend),
            "selected_case_threshold": float(selected_case_threshold),
            "base_best_iteration": int(base_best_it),
            "tail_best_iteration": int(tail_best_it),
            "base_best_score": float(base_best_score),
            "tail_best_score": float(tail_best_score),
            "router_val_case_accuracy": case_accuracy_val,
            "router_val_case_accuracy_runtime": case_accuracy_val_runtime,
            "router_test_case_accuracy": case_accuracy_test,
            "router_best_iteration": int(selected_router_train_artifact.best_iteration),
            "router_trials": int(MOE_ROUTER_TRIALS),
            "router_optuna_best_score": (
                float(selected_router_optuna_score)
                if selected_router_optuna_score is not None
                else None
            ),
            "router_optuna_best_blend": (
                float(selected_router_optuna_best_blend)
                if selected_router_optuna_best_blend is not None
                else None
            ),
            "router_best_params": {
                k: (float(v) if isinstance(v, float) else int(v))
                for k, v in selected_router_params.items()
            },
            "calibration_grid": calibration_results,
        },
        "validation": val_metrics,
        "test": test_metrics,
        "artifacts": {
            "base_train_model": CLASSIFIER_XGB_MOE_BASE_TRAIN_PATH,
            "base_prod_model": CLASSIFIER_XGB_MOE_BASE_PATH,
            "tail_train_model": CLASSIFIER_XGB_MOE_TAIL_TRAIN_PATH,
            "tail_prod_model": CLASSIFIER_XGB_MOE_TAIL_PATH,
            "router_train_model": CLASSIFIER_XGB_MOE_ROUTER_TRAIN_PATH,
            "router_prod_model": CLASSIFIER_XGB_MOE_ROUTER_PATH,
            "base_local_to_global": {
                str(k): CLASSIFIER_LABEL_LIST[v]
                for k, v in sorted(base_local_to_global.items())
            },
        },
    }
    os.makedirs(EVAL_METRICS_DIR, exist_ok=True)
    metrics_path = os.path.join(EVAL_METRICS_DIR, "classifier_xgb_moe_metrics.yaml")
    with open(metrics_path, "w", encoding="utf-8") as f:
        _ = yaml.safe_dump(metrics, f, sort_keys=False)
    print(f"[moe] metrics written to {metrics_path}")


def main() -> None:
    """Main training function for XGBoost classifier."""
    np.random.seed(SEED)
    if TRAINING_MODE not in {"single", "moe"}:
        raise ValueError(
            "PAGE_CLASSIFIER_XGB_MODE must be one of {'single', 'moe'}."
        )
    year_window = 5
    length_bucket_edges = [0.0, 120, 130, 200, float("inf")]
    back_matter_bucket_edges = [0.0, 35, 60, 105, float("inf")]
    data_path = os.path.join(DATA_DIR, "page-data.parquet")
    split_path = os.path.join(DATA_DIR, "agreement-splits.json")
    if TRAINING_MODE == "moe":
        _run_moe_training(
            data_path=data_path,
            split_path=split_path,
            year_window=year_window,
            length_bucket_edges=length_bucket_edges,
            back_matter_bucket_edges=back_matter_bucket_edges,
        )
        return

    # Load and prepare data
    features, feature_names, y, agreement_uuids, years, _orders, split_df = load_and_prepare_data(
        data_path
    )
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
    if features.shape[1] != len(feature_names):
        raise ValueError("Feature matrix width does not match extracted feature names.")

    # Train/val/test split from shared manifest (agreement-level, year-stratified)
    agreement_ids = np.asarray(pd.unique(agreement_uuids), dtype=str)
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

    agreement_windows = cast(
        pd.Series,
        pd.DataFrame(
            {"agreement_uuid": agreement_uuids, "announcement_year": years}
        )
        .drop_duplicates(subset=["agreement_uuid"])
        .set_index("agreement_uuid")["announcement_year"]
        .loc[agreement_ids],
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
    agreement_stats = cast(
        pd.DataFrame,
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
        buckets = cast(
            pd.Series, pd.cut(values, bins=edges, include_lowest=True, labels=False)
        )
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
                cast(pd.Series, agreement_stats.loc[list(ids), "page_count"]),
                length_edges,
            )
        if back_edges:
            back_counts = _bucket_counts(
                cast(pd.Series, agreement_stats.loc[list(ids), "back_count"]),
                back_edges,
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
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

    num_classes = len(CLASSIFIER_LABEL_LIST)
    back_idx = CLASSIFIER_LABEL_LIST.index("back_matter")

    hard_negative_train_mask: np.ndarray | None = None
    hard_negative_train_full_mask: np.ndarray | None = None

    # Hyperparameter optimization
    if USE_HARD_NEGATIVES:
        hard_negative_train_mask = hard_negative_back_matter_mask(
            X_train,
            y_train,
            feature_names,
            back_label_idx=back_idx,
        )
        hard_negative_train_full_mask = hard_negative_back_matter_mask(
            np.asarray(features[train_mask | val_model_mask]),
            np.asarray(y[train_mask | val_model_mask]),
            feature_names,
            back_label_idx=back_idx,
        )
    print(
        f"Starting hyperparameter optimization ({OPTUNA_TRIALS} trials, neighbor_context={USE_NEIGHBOR_CONTEXT}, hard_negatives={USE_HARD_NEGATIVES})..."
    )
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))

    # Define objective with fixed data
    def objective_wrapper(trial: optuna.Trial) -> float:
        """Wrapper for objective function with fixed data."""
        return objective(
            trial,
            X_train,
            y_train,
            X_val_model,
            y_val_model,
            feature_names,
            hard_negative_train_mask,
            USE_HARD_NEGATIVES,
        )

    study.optimize(objective_wrapper, n_trials=OPTUNA_TRIALS)
    print("Best hyperparameters:", study.best_params)

    best_it = int(study.best_trial.user_attrs.get("best_iteration", NUM_BOOST_ROUND))
    best_params = study.best_params.copy()
    sig_weight_scale = float(best_params.pop("sig_weight_scale"))
    back_weight_scale = float(best_params.pop("back_weight_scale"))
    hard_negative_scale = float(best_params.pop("hard_negative_scale", 1.0))
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
    class_w_train = _class_weight_vector(
        y_train,
        num_classes,
        sig_weight_scale=sig_weight_scale,
        back_weight_scale=back_weight_scale,
    )
    class_w_full = _class_weight_vector(
        y_train_full,
        num_classes,
        sig_weight_scale=sig_weight_scale,
        back_weight_scale=back_weight_scale,
    )
    train_weights = class_w_train[y_train].astype(np.float32, copy=False)
    train_full_weights = class_w_full[y_train_full].astype(np.float32, copy=False)
    if USE_HARD_NEGATIVES:
        if hard_negative_train_mask is None or hard_negative_train_full_mask is None:
            raise ValueError("Hard-negative masks were not prepared.")
        if hard_negative_train_mask.shape[0] != y_train.shape[0]:
            raise ValueError("hard_negative_train_mask length mismatch.")
        if hard_negative_train_full_mask.shape[0] != y_train_full.shape[0]:
            raise ValueError("hard_negative_train_full_mask length mismatch.")
        hard_mult_train = np.ones_like(train_weights, dtype=np.float32)
        hard_mult_train[hard_negative_train_mask] = hard_negative_scale
        train_weights = train_weights * hard_mult_train
        hard_mult_full = np.ones_like(train_full_weights, dtype=np.float32)
        hard_mult_full[hard_negative_train_full_mask] = hard_negative_scale
        train_full_weights = train_full_weights * hard_mult_full
    dtrain = xgb.DMatrix(
        X_train,
        label=y_train,
        weight=train_weights,
        feature_names=feature_names,
    )
    dtrain_full = xgb.DMatrix(
        X_train_full,
        label=y_train_full,
        weight=train_full_weights,
        feature_names=feature_names,
    )

    train_only_model = xgb.train(best_params, dtrain, num_boost_round=best_it)
    train_model_path = CLASSIFIER_XGB_TRAIN_PATH
    train_only_model.save_model(train_model_path)
    print(f"Train-only model saved to {train_model_path}")

    final_model = xgb.train(best_params, dtrain_full, num_boost_round=best_it)
    model_path = CLASSIFIER_XGB_PATH
    final_model.save_model(model_path)
    print(f"Production model saved to {model_path}")

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
    feature_importance_gain = final_model.get_score(importance_type="gain")
    feature_importance_weight = final_model.get_score(importance_type="weight")
    def _importance_to_float(value: float | list[float]) -> float:
        if isinstance(value, list):
            if not value:
                return 0.0
            return float(np.mean(np.asarray(value, dtype=float)))
        return float(value)

    def _importance_to_int(value: float | list[float]) -> int:
        if isinstance(value, list):
            if not value:
                return 0
            return int(np.sum(np.asarray(value, dtype=float)))
        return int(value)

    top_feature_importance_gain = {
        name: _importance_to_float(value)
        for name, value in sorted(
            feature_importance_gain.items(),
            key=lambda item: _importance_to_float(item[1]),
            reverse=True,
        )[:25]
    }
    top_feature_importance_weight = {
        name: _importance_to_int(value)
        for name, value in sorted(
            feature_importance_weight.items(),
            key=lambda item: _importance_to_int(item[1]),
            reverse=True,
        )[:25]
    }
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
        "optimization": {
            "objective": "sig_critical_score",
            "use_neighbor_context": bool(USE_NEIGHBOR_CONTEXT),
            "use_hard_negatives": bool(USE_HARD_NEGATIVES),
            "sig_weight_scale": sig_weight_scale,
            "back_weight_scale": back_weight_scale,
            "hard_negative_scale": hard_negative_scale,
            "best_iteration": int(best_it),
            "best_sig_critical_score": float(study.best_value),
            "best_val_macro_f1": float(study.best_trial.user_attrs.get("val_macro_f1", 0.0)),
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
        "feature_importance_gain_top25": top_feature_importance_gain,
        "feature_importance_weight_top25": top_feature_importance_weight,
    }
    os.makedirs(EVAL_METRICS_DIR, exist_ok=True)
    metrics_path = os.path.join(EVAL_METRICS_DIR, "classifier_xgb_test_metrics.yaml")
    with open(metrics_path, "w", encoding="utf-8") as f:
        _ = yaml.safe_dump(metrics, f, sort_keys=False)
    print(f"Test metrics written to {metrics_path}")

if __name__ == "__main__":
    main()
