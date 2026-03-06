"""Agreement-level review model for page classifier QA."""
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
import importlib
from typing import Protocol, TypedDict, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .crf_pipeline import (
    AgreementDocument,
    CRFHyperparameters,
    CRFModelProtocol,
    DEFAULT_DATA_PATH,
    DEFAULT_OPTUNA_PATH,
    DEFAULT_SPLIT_PATH,
    build_agreement_documents,
    build_feature_sequences,
    build_label_sequences,
    enforce_monotonic_transition_weights,
    fit_tfidf_vectorizer,
    load_or_create_split_manifest,
    load_page_dataframe,
    postprocess_prediction_sequences,
    split_dataframe_from_manifest,
)
from .inference import (
    DEFAULT_REVIEW_LOW_CONFIDENCE_THRESHOLD,
    DEFAULT_REVIEW_LOW_MARGIN_THRESHOLD,
    DEFAULT_REVIEW_VERY_LOW_CONFIDENCE_THRESHOLD,
    AgreementReviewSummary,
    ClassifierInference,
    build_agreement_review_summary,
    review_summaries_to_frame,
)
from .page_classifier_constants import (
    CLASSIFIER_CRF_TUNE_VAL_PATH,
    CLASSIFIER_REVIEW_METRICS_PATH,
    CLASSIFIER_REVIEW_MODEL_PATH,
)

DEFAULT_REVIEW_OOF_SPLITS = 5
DEFAULT_ILLEGAL_TRANSITION_WEIGHT = -1000.0
DEFAULT_REVIEW_THRESHOLD = 0.5
DEFAULT_TARGET_REVIEW_RECALL = 0.98
DEFAULT_REVIEW_RANKING_CUTOFFS = (10, 20, 30, 40, 50, 60, 70, 80, 100)
DEFAULT_REVIEW_TRAIN_OOF_CACHE_PATH = (
    Path(__file__).resolve().parent / "eval_metrics" / "page_classifier_review_train_oof_features.csv"
)
DEFAULT_REVIEW_VAL_CACHE_PATH = (
    Path(__file__).resolve().parent / "eval_metrics" / "page_classifier_review_val_features.csv"
)


def _log(message: str) -> None:
    print(message, flush=True)


class ReviewPrediction(TypedDict):
    agreement_uuid: str
    needs_review: bool
    review_probability: float
    review_threshold: float
    review_score: float


class ReviewArtifact(TypedDict):
    model: Pipeline
    model_name: str
    feature_columns: list[str]
    threshold: float
    page_model_hyperparameters: CRFHyperparameters
    summary_thresholds: dict[str, float]


class ReviewCRFProtocol(Protocol):
    def fit(
        self,
        x_train: list[list[dict[str, float | bool]]],
        y_train: list[list[str]],
    ) -> object: ...

    def predict(self, x_test: list[list[dict[str, float | bool]]]) -> list[list[str]]: ...

    def predict_marginals(
        self,
        x_test: list[list[dict[str, float | bool]]],
    ) -> list[list[dict[object, object]]]: ...

    transition_features_: dict[tuple[str, str], float]


class ReviewCRFSuiteModuleProtocol(Protocol):
    def CRF(
        self,
        *,
        algorithm: str,
        c1: float,
        c2: float,
        max_iterations: int,
        all_possible_transitions: bool,
    ) -> ReviewCRFProtocol: ...


@dataclass(frozen=True)
class ReviewDataset:
    frame: pd.DataFrame
    feature_columns: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the agreement-level review model from CRF uncertainty features."
    )
    _ = parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Parquet dataset path.",
    )
    _ = parser.add_argument(
        "--split-path",
        type=Path,
        default=DEFAULT_SPLIT_PATH,
        help="Stable train/val/test split manifest.",
    )
    _ = parser.add_argument(
        "--optuna-path",
        type=Path,
        default=DEFAULT_OPTUNA_PATH,
        help="Best page-model Optuna summary used to recover CRF hyperparameters.",
    )
    _ = parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(CLASSIFIER_REVIEW_MODEL_PATH),
        help="Where to save the trained review-model artifact.",
    )
    _ = parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path(CLASSIFIER_REVIEW_METRICS_PATH),
        help="Where to save structured review-model metrics.",
    )
    _ = parser.add_argument(
        "--page-model-path",
        type=Path,
        default=Path(CLASSIFIER_CRF_TUNE_VAL_PATH),
        help="Optional existing CRF artifact for review-model inference only.",
    )
    _ = parser.add_argument(
        "--train-oof-cache-path",
        type=Path,
        default=DEFAULT_REVIEW_TRAIN_OOF_CACHE_PATH,
        help="CSV path for cached train OOF agreement-level review features.",
    )
    _ = parser.add_argument(
        "--val-cache-path",
        type=Path,
        default=DEFAULT_REVIEW_VAL_CACHE_PATH,
        help="CSV path for cached validation agreement-level review features.",
    )
    _ = parser.add_argument(
        "--rebuild-feature-cache",
        action="store_true",
        help="Ignore existing cached review feature CSVs and rebuild them from CRF predictions.",
    )
    _ = parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of split buckets in the existing agreement manifest.",
    )
    _ = parser.add_argument(
        "--fold-index",
        type=int,
        default=0,
        help="Test fold index used by the page-classifier split manifest.",
    )
    _ = parser.add_argument(
        "--val-fold-index",
        type=int,
        default=1,
        help="Validation fold index used by the page-classifier split manifest.",
    )
    _ = parser.add_argument(
        "--split-seed",
        type=int,
        default=2718,
        help="Split seed used by the page-classifier split manifest.",
    )
    _ = parser.add_argument(
        "--oof-splits",
        type=int,
        default=DEFAULT_REVIEW_OOF_SPLITS,
        help="Number of out-of-fold train splits used to build review-model training features.",
    )
    _ = parser.add_argument(
        "--illegal-transition-weight",
        type=float,
        default=DEFAULT_ILLEGAL_TRANSITION_WEIGHT,
        help="Weight assigned to forbidden backward transitions in the CRF.",
    )
    return parser.parse_args()


def _save_metrics(metrics_path: Path, payload: dict[str, object]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    _ = metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _review_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns.tolist()
        if column not in {"agreement_uuid", "page_error_count", "needs_review", "exact_match"}
    ]


def _coerce_cached_review_frame(frame: pd.DataFrame) -> pd.DataFrame:
    expected_bool_columns = {"needs_review", "exact_match"}
    expected_string_columns = {"agreement_uuid"}
    working = frame.copy()
    missing_columns = {"agreement_uuid", "page_error_count", "needs_review", "exact_match"} - set(working.columns)
    if missing_columns:
        raise ValueError(f"Cached review feature CSV missing columns: {sorted(missing_columns)}")
    for column in working.columns.tolist():
        if column in expected_string_columns:
            working[column] = working[column].astype(str)
        elif column in expected_bool_columns:
            working[column] = working[column].astype(bool)
        else:
            working[column] = pd.to_numeric(working[column], errors="raise")
    return working


def _load_cached_review_dataset(cache_path: Path) -> ReviewDataset:
    frame = pd.read_csv(cache_path)
    coerced_frame = _coerce_cached_review_frame(frame)
    return ReviewDataset(frame=coerced_frame, feature_columns=_review_feature_columns(coerced_frame))


def _save_cached_review_dataset(dataset: ReviewDataset, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.frame.to_csv(cache_path, index=False)


def _load_page_model_hyperparameters(optuna_path: Path) -> CRFHyperparameters:
    payload = json.loads(optuna_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Optuna summary must be a mapping.")
    best_params = payload.get("best_params")
    if not isinstance(best_params, dict):
        raise ValueError("Optuna summary missing `best_params`.")
    c1_value = best_params.get("c1")
    c2_value = best_params.get("c2")
    tfidf_value = best_params.get("tfidf_max_features")
    if isinstance(c1_value, bool) or not isinstance(c1_value, (int, float)):
        raise ValueError("Optuna best param `c1` must be numeric.")
    if isinstance(c2_value, bool) or not isinstance(c2_value, (int, float)):
        raise ValueError("Optuna best param `c2` must be numeric.")
    if isinstance(tfidf_value, bool) or not isinstance(tfidf_value, int):
        raise ValueError("Optuna best param `tfidf_max_features` must be an integer.")
    return {
        "c1": float(c1_value),
        "c2": float(c2_value),
        "tfidf_max_features": int(tfidf_value),
    }


def _load_review_crf_module() -> ReviewCRFSuiteModuleProtocol:
    try:
        return cast(
            ReviewCRFSuiteModuleProtocol,
            cast(object, importlib.import_module("sklearn_crfsuite")),
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing CRF dependency: install `sklearn-crfsuite` in the ETL environment "
            + "before training the review model."
        ) from exc


def _build_review_rows(
    documents: list[AgreementDocument],
    predicted_labels: list[list[str]],
    marginal_rows: list[list[dict[str, float]]],
) -> pd.DataFrame:
    summaries: list[AgreementReviewSummary] = []
    page_error_counts: list[int] = []
    for document, predicted_sequence, marginal_sequence in zip(
        documents,
        predicted_labels,
        marginal_rows,
        strict=True,
    ):
        page_error_count = sum(
            true_label != predicted_label
            for true_label, predicted_label in zip(
                document.labels,
                predicted_sequence,
                strict=True,
            )
        )
        page_error_counts.append(page_error_count)
        summaries.append(
            build_agreement_review_summary(
                document.agreement_uuid,
                predicted_sequence,
                marginal_sequence,
                low_confidence_threshold=DEFAULT_REVIEW_LOW_CONFIDENCE_THRESHOLD,
                very_low_confidence_threshold=DEFAULT_REVIEW_VERY_LOW_CONFIDENCE_THRESHOLD,
                low_margin_threshold=DEFAULT_REVIEW_LOW_MARGIN_THRESHOLD,
            )
        )

    frame = review_summaries_to_frame(summaries)
    frame["page_error_count"] = [float(value) for value in page_error_counts]
    frame["needs_review"] = [bool(value > 0) for value in page_error_counts]
    frame["exact_match"] = [bool(value == 0) for value in page_error_counts]
    return frame


def _train_and_predict_crf(
    *,
    train_documents: list[AgreementDocument],
    eval_documents: list[AgreementDocument],
    page_model_hyperparameters: CRFHyperparameters,
    illegal_transition_weight: float,
    log_prefix: str | None = None,
) -> tuple[list[list[str]], list[list[dict[str, float]]]]:
    prefix = f"{log_prefix} " if log_prefix else ""
    train_page_texts = [page_text for doc in train_documents for page_text in doc.page_texts]
    _log(
        f"{prefix}[review] preparing CRF features: train_agreements={len(train_documents)}, "
        + f"eval_agreements={len(eval_documents)}"
    )
    vectorizer = fit_tfidf_vectorizer(
        train_page_texts,
        max_features=int(page_model_hyperparameters["tfidf_max_features"]),
    )
    x_train = build_feature_sequences(train_documents, vectorizer)
    y_train = build_label_sequences(train_documents)
    x_eval = build_feature_sequences(eval_documents, vectorizer)

    sklearn_crfsuite = _load_review_crf_module()
    crf_model = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=float(page_model_hyperparameters["c1"]),
        c2=float(page_model_hyperparameters["c2"]),
        max_iterations=100,
        all_possible_transitions=True,
    )
    _log(
        f"{prefix}[review] fitting CRF: c1={float(page_model_hyperparameters['c1']):.6f}, "
        + f"c2={float(page_model_hyperparameters['c2']):.6f}, "
        + f"tfidf_max_features={int(page_model_hyperparameters['tfidf_max_features'])}"
    )
    _ = crf_model.fit(x_train, y_train)
    _ = enforce_monotonic_transition_weights(
        cast(CRFModelProtocol, cast(object, crf_model)),
        illegal_transition_weight=illegal_transition_weight,
    )

    _log(f"{prefix}[review] predicting eval fold")
    predicted_raw = crf_model.predict(x_eval)
    marginals_raw = crf_model.predict_marginals(x_eval)
    predicted_raw_labels = [[str(label) for label in sequence] for sequence in predicted_raw]
    predicted_labels, modified_agreement_count, modified_page_count = postprocess_prediction_sequences(
        predicted_raw_labels,
        x_eval,
    )
    if modified_page_count > 0:
        _log(
            f"{prefix}[review] postprocess adjusted {modified_page_count} pages across "
            + f"{modified_agreement_count} agreements"
        )
    marginal_rows = [
        [{str(label): float(cast(float, prob)) for label, prob in row.items()} for row in sequence]
        for sequence in marginals_raw
    ]
    return predicted_labels, marginal_rows


def _shuffled_document_indices(documents: list[AgreementDocument], *, seed: int) -> list[int]:
    indices = list(range(len(documents)))
    random.Random(seed).shuffle(indices)
    return indices


def _build_review_training_dataset(
    *,
    train_documents: list[AgreementDocument],
    val_documents: list[AgreementDocument],
    page_model_hyperparameters: CRFHyperparameters,
    illegal_transition_weight: float,
    oof_splits: int,
    split_seed: int,
) -> tuple[ReviewDataset, ReviewDataset]:
    if len(train_documents) < 2:
        raise ValueError("Need at least two agreements to build a review-model training set.")

    shuffled_indices = _shuffled_document_indices(train_documents, seed=split_seed)
    shuffled_documents = [train_documents[index] for index in shuffled_indices]
    groups = np.asarray([doc.agreement_uuid for doc in shuffled_documents], dtype=object)
    n_splits = min(oof_splits, len(shuffled_documents))
    if n_splits < 2:
        raise ValueError("oof_splits must resolve to at least 2.")

    oof_frames: list[pd.DataFrame] = []
    splitter = GroupKFold(n_splits=n_splits)
    sample_indices = np.arange(len(shuffled_documents), dtype=np.int64)
    for fold_number, (fold_train_indices, fold_eval_indices) in enumerate(
        splitter.split(sample_indices, groups=groups),
        start=1,
    ):
        fold_train_documents = [shuffled_documents[int(index)] for index in fold_train_indices.tolist()]
        fold_eval_documents = [shuffled_documents[int(index)] for index in fold_eval_indices.tolist()]
        _log(
            f"[review][oof {fold_number}/{n_splits}] "
            + f"train_agreements={len(fold_train_documents)} eval_agreements={len(fold_eval_documents)}"
        )
        fold_predictions, fold_marginals = _train_and_predict_crf(
            train_documents=fold_train_documents,
            eval_documents=fold_eval_documents,
            page_model_hyperparameters=page_model_hyperparameters,
            illegal_transition_weight=illegal_transition_weight,
            log_prefix=f"[review][oof {fold_number}/{n_splits}]",
        )
        oof_frames.append(
            _build_review_rows(
                fold_eval_documents,
                fold_predictions,
                fold_marginals,
            )
        )
    train_frame = pd.concat(oof_frames, ignore_index=True)
    train_frame = train_frame.sort_values("agreement_uuid", kind="mergesort").reset_index(drop=True)

    val_predictions, val_marginals = _train_and_predict_crf(
        train_documents=train_documents,
        eval_documents=val_documents,
        page_model_hyperparameters=page_model_hyperparameters,
        illegal_transition_weight=illegal_transition_weight,
        log_prefix="[review][val-crf]",
    )
    val_frame = _build_review_rows(val_documents, val_predictions, val_marginals)
    val_frame = val_frame.sort_values("agreement_uuid", kind="mergesort").reset_index(drop=True)
    feature_columns = _review_feature_columns(train_frame)
    return ReviewDataset(frame=train_frame, feature_columns=feature_columns), ReviewDataset(
        frame=val_frame,
        feature_columns=feature_columns,
    )

def _build_review_model_candidates() -> dict[str, Pipeline]:
    return {
        "logreg_balanced_l2_c1.0": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=2718,
                        C=1.0,
                    ),
                ),
            ]
        ),
        "logreg_balanced_l2_c0.5": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=2718,
                        C=0.5,
                    ),
                ),
            ]
        ),
        "logreg_balanced_l2_c0.25": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=2718,
                        C=0.25,
                    ),
                ),
            ]
        ),
        "logreg_balanced_l2_c0.15": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=2718,
                        C=0.15,
                    ),
                ),
            ]
        ),
        "logreg_balanced_l2_c0.10": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=2718,
                        C=0.10,
                    ),
                ),
            ]
        ),
        "logreg_balanced_l2_c0.05": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=2718,
                        C=0.05,
                    ),
                ),
            ]
        ),
        "logreg_unweighted_l2_c1.0": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        max_iter=5000,
                        random_state=2718,
                        C=1.0,
                    ),
                ),
            ]
        ),
        "logreg_unweighted_l2_c0.5": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        max_iter=5000,
                        random_state=2718,
                        C=0.5,
                    ),
                ),
            ]
        ),
        "logreg_unweighted_l2_c0.25": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        max_iter=5000,
                        random_state=2718,
                        C=0.25,
                    ),
                ),
            ]
        ),
        "logreg_unweighted_l2_c0.15": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        max_iter=5000,
                        random_state=2718,
                        C=0.15,
                    ),
                ),
            ]
        ),
        "logreg_unweighted_l2_c0.10": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        max_iter=5000,
                        random_state=2718,
                        C=0.10,
                    ),
                ),
            ]
        ),
        "logreg_unweighted_l2_c0.05": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        max_iter=5000,
                        random_state=2718,
                        C=0.05,
                    ),
                ),
            ]
        ),
        "gbdt_depth2_lr0.05_n100": Pipeline(
            steps=[
                ("scaler", "passthrough"),
                (
                    "gbdt",
                    GradientBoostingClassifier(
                        learning_rate=0.05,
                        max_depth=2,
                        min_samples_leaf=5,
                        n_estimators=100,
                        random_state=2718,
                        subsample=0.8,
                    ),
                ),
            ]
        ),
        "gbdt_depth2_lr0.10_n100": Pipeline(
            steps=[
                ("scaler", "passthrough"),
                (
                    "gbdt",
                    GradientBoostingClassifier(
                        learning_rate=0.10,
                        max_depth=2,
                        min_samples_leaf=5,
                        n_estimators=100,
                        random_state=2718,
                        subsample=0.8,
                    ),
                ),
            ]
        ),
        "gbdt_depth3_lr0.05_n150": Pipeline(
            steps=[
                ("scaler", "passthrough"),
                (
                    "gbdt",
                    GradientBoostingClassifier(
                        learning_rate=0.05,
                        max_depth=3,
                        min_samples_leaf=5,
                        n_estimators=150,
                        random_state=2718,
                        subsample=0.8,
                    ),
                ),
            ]
        ),
        "gbdt_depth3_lr0.10_n100": Pipeline(
            steps=[
                ("scaler", "passthrough"),
                (
                    "gbdt",
                    GradientBoostingClassifier(
                        learning_rate=0.10,
                        max_depth=3,
                        min_samples_leaf=5,
                        n_estimators=100,
                        random_state=2718,
                        subsample=0.8,
                    ),
                ),
            ]
        ),
    }


def _select_best_review_model(
    train_dataset: ReviewDataset,
    val_dataset: ReviewDataset,
) -> tuple[str, Pipeline, dict[str, float | int], list[dict[str, object]]]:
    x_train = train_dataset.frame[train_dataset.feature_columns]
    y_train = train_dataset.frame["needs_review"].astype(int)
    x_val = val_dataset.frame[val_dataset.feature_columns]
    y_val = val_dataset.frame["needs_review"].astype(int).to_numpy(dtype=np.int64)

    best_model_name: str | None = None
    best_model: Pipeline | None = None
    best_threshold_metrics: dict[str, float | int] | None = None
    candidate_summaries: list[dict[str, object]] = []
    for model_name, candidate_model in _build_review_model_candidates().items():
        _log(f"[review] fitting candidate review model: {model_name}")
        _ = candidate_model.fit(x_train, y_train)
        val_probabilities = cast(np.ndarray, candidate_model.predict_proba(x_val))[:, 1]
        threshold_metrics = _select_review_threshold(y_val, val_probabilities)
        candidate_summary: dict[str, object] = {
            "model_name": model_name,
            **threshold_metrics,
        }
        candidate_summaries.append(candidate_summary)
        _log(
            "[review] candidate result: "
            + f"{model_name} threshold={float(threshold_metrics['threshold']):.6f} "
            + f"flagged={int(threshold_metrics['flagged_count'])} "
            + f"precision={float(threshold_metrics['precision']):.4f} "
            + f"recall={float(threshold_metrics['recall']):.4f}"
        )
        if best_model_name is None or (
            cast(int, threshold_metrics["meets_target_recall"]) * -1,
            cast(int, threshold_metrics["flagged_count"]),
            -cast(float, threshold_metrics["precision"]),
            -cast(float, threshold_metrics["f1"]),
            -cast(float, threshold_metrics["threshold"]),
        ) < (
            cast(int, cast(dict[str, float | int], best_threshold_metrics)["meets_target_recall"]) * -1,
            cast(int, cast(dict[str, float | int], best_threshold_metrics)["flagged_count"]),
            -cast(float, cast(dict[str, float | int], best_threshold_metrics)["precision"]),
            -cast(float, cast(dict[str, float | int], best_threshold_metrics)["f1"]),
            -cast(float, cast(dict[str, float | int], best_threshold_metrics)["threshold"]),
        ):
            best_model_name = model_name
            best_model = candidate_model
            best_threshold_metrics = threshold_metrics

    if best_model_name is None or best_model is None or best_threshold_metrics is None:
        raise RuntimeError("Failed to fit any review-model candidates.")
    return best_model_name, best_model, best_threshold_metrics, candidate_summaries


def _extract_model_feature_rows(
    model: Pipeline,
    feature_columns: list[str],
) -> tuple[str, list[dict[str, str | float]], float | None]:
    if "logreg" in model.named_steps:
        coefficients = cast(
            np.ndarray,
            cast(LogisticRegression, model.named_steps["logreg"]).coef_,
        )
        intercept = cast(
            np.ndarray,
            cast(LogisticRegression, model.named_steps["logreg"]).intercept_,
        )
        rows = sorted(
            [
                {
                    "feature": feature_name,
                    "coefficient": float(coefficients[0, feature_index]),
                }
                for feature_index, feature_name in enumerate(feature_columns)
            ],
            key=lambda row: abs(cast(float, row["coefficient"])),
            reverse=True,
        )
        return "coefficient", rows, float(intercept[0])
    if "gbdt" in model.named_steps:
        importances = cast(
            np.ndarray,
            cast(GradientBoostingClassifier, model.named_steps["gbdt"]).feature_importances_,
        )
        rows = sorted(
            [
                {
                    "feature": feature_name,
                    "importance": float(importances[feature_index]),
                }
                for feature_index, feature_name in enumerate(feature_columns)
            ],
            key=lambda row: cast(float, row["importance"]),
            reverse=True,
        )
        return "importance", rows, None
    raise ValueError(f"Unsupported review model pipeline steps: {list(model.named_steps.keys())}")


def _select_review_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    target_recall: float = DEFAULT_TARGET_REVIEW_RECALL,
) -> dict[str, float | int]:
    candidate_thresholds = sorted({float(prob) for prob in y_prob.tolist()}, reverse=True)
    if 0.0 not in candidate_thresholds:
        candidate_thresholds.append(0.0)

    best_feasible_metrics: dict[str, float | int] | None = None
    best_fallback_metrics: dict[str, float | int] | None = None
    for threshold in candidate_thresholds:
        y_pred = y_prob >= threshold
        precision, recall, f1, _ = cast(
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            precision_recall_fscore_support(
                y_true,
                y_pred.astype(int),
                labels=[1],
                average=None,
                zero_division=cast(str, cast(object, 0)),
            ),
        )
        flagged_count = int(np.sum(y_pred))
        candidate = {
            "threshold": float(threshold),
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "flagged_count": flagged_count,
        }
        if cast(float, candidate["recall"]) >= target_recall:
            if best_feasible_metrics is None or (
                cast(int, candidate["flagged_count"]),
                -cast(float, candidate["precision"]),
                -cast(float, candidate["f1"]),
                -cast(float, candidate["threshold"]),
            ) < (
                cast(int, best_feasible_metrics["flagged_count"]),
                -cast(float, best_feasible_metrics["precision"]),
                -cast(float, best_feasible_metrics["f1"]),
                -cast(float, best_feasible_metrics["threshold"]),
            ):
                best_feasible_metrics = candidate
            continue
        if best_fallback_metrics is None or (
            cast(float, candidate["recall"]),
            cast(float, candidate["precision"]),
            cast(float, candidate["f1"]),
            -cast(int, candidate["flagged_count"]),
            cast(float, candidate["threshold"]),
        ) > (
            cast(float, best_fallback_metrics["recall"]),
            cast(float, best_fallback_metrics["precision"]),
            cast(float, best_fallback_metrics["f1"]),
            -cast(int, best_fallback_metrics["flagged_count"]),
            cast(float, best_fallback_metrics["threshold"]),
        ):
            best_fallback_metrics = candidate

    if best_feasible_metrics is not None:
        return {
            **best_feasible_metrics,
            "target_recall": target_recall,
            "meets_target_recall": 1,
        }
    if best_fallback_metrics is None:
        return {
            "threshold": DEFAULT_REVIEW_THRESHOLD,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "flagged_count": 0,
            "target_recall": target_recall,
            "meets_target_recall": 0,
        }
    return {
        **best_fallback_metrics,
        "target_recall": target_recall,
        "meets_target_recall": 0,
    }


def _build_ranking_metrics(
    *,
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    cutoffs: tuple[int, ...] = DEFAULT_REVIEW_RANKING_CUTOFFS,
) -> dict[str, object]:
    ranking_rows = [
        {
            "agreement_uuid": str(row["agreement_uuid"]),
            "needs_review": bool(row["needs_review"]),
            "page_error_count": int(float(row["page_error_count"])),
            "review_probability": float(probability),
        }
        for row, probability in zip(frame.to_dict("records"), probabilities.tolist(), strict=True)
    ]
    ranking_rows.sort(
        key=lambda row: (
            -cast(float, row["review_probability"]),
            cast(str, row["agreement_uuid"]),
        )
    )
    positive_count = sum(1 for row in ranking_rows if bool(row["needs_review"]))
    metrics_by_cutoff: dict[str, dict[str, float | int]] = {}
    for cutoff in cutoffs:
        capped_cutoff = min(cutoff, len(ranking_rows))
        if capped_cutoff <= 0:
            continue
        top_rows = ranking_rows[:capped_cutoff]
        hits = sum(1 for row in top_rows if bool(row["needs_review"]))
        metrics_by_cutoff[str(cutoff)] = {
            "top_k": capped_cutoff,
            "hits": hits,
            "precision_at_k": float(hits) / float(capped_cutoff),
            "recall_at_k": float(hits) / float(positive_count) if positive_count > 0 else 0.0,
        }
    return {
        "positive_agreement_count": positive_count,
        "cutoffs": metrics_by_cutoff,
    }


def _evaluate_review_model(
    *,
    name: str,
    dataset: ReviewDataset,
    model: Pipeline,
    threshold: float,
) -> dict[str, object]:
    x_eval = dataset.frame[dataset.feature_columns]
    y_true = dataset.frame["needs_review"].astype(int).to_numpy(dtype=np.int64)
    y_prob = model.predict_proba(x_eval)[:, 1]
    y_pred = y_prob >= threshold
    precision, recall, f1, support = cast(
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        precision_recall_fscore_support(
            y_true,
            y_pred.astype(int),
            labels=[0, 1],
            average=None,
            zero_division=cast(str, cast(object, 0)),
        ),
    )
    metrics: dict[str, object] = {
        "dataset": name,
        "agreement_count": int(len(dataset.frame)),
        "positive_agreement_count": int(np.sum(y_true)),
        "flagged_agreement_count": int(np.sum(y_pred)),
        "threshold": float(threshold),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "class_metrics": {
            "clean": {
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(f1[0]),
                "support": int(support[0]),
            },
            "needs_review": {
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1": float(f1[1]),
                "support": int(support[1]),
            },
        },
        "ranking_metrics": _build_ranking_metrics(frame=dataset.frame, probabilities=y_prob),
        "examples": [
            {
                "agreement_uuid": str(row["agreement_uuid"]),
                "needs_review": bool(row["needs_review"]),
                "review_probability": float(prob),
                "predicted_needs_review": bool(pred),
                "page_error_count": int(float(row["page_error_count"])),
            }
            for row, prob, pred in zip(
                dataset.frame.to_dict("records"),
                y_prob.tolist(),
                y_pred.tolist(),
                strict=True,
            )
        ],
    }
    return metrics


def train_review_model(
    *,
    data_path: Path,
    split_path: Path,
    optuna_path: Path,
    model_path: Path,
    metrics_path: Path,
    train_oof_cache_path: Path,
    val_cache_path: Path,
    rebuild_feature_cache: bool,
    n_splits: int,
    fold_index: int,
    val_fold_index: int | None,
    split_seed: int,
    oof_splits: int,
    illegal_transition_weight: float,
) -> dict[str, object]:
    df = load_page_dataframe(data_path)
    _log(f"[review] loaded dataset: rows={len(df)}, agreements={df['agreement_uuid'].nunique()}")
    split_manifest = load_or_create_split_manifest(
        df,
        split_path=split_path,
        n_splits=n_splits,
        test_fold_index=fold_index,
        val_fold_index=val_fold_index,
        split_seed=split_seed,
    )
    train_df, val_df, _ = split_dataframe_from_manifest(df, split_manifest)
    _log(
        "[review] split counts: "
        + f"train={train_df['agreement_uuid'].nunique()}, "
        + f"val={val_df['agreement_uuid'].nunique()}, "
        + f"oof_splits={oof_splits}"
    )
    train_documents = build_agreement_documents(train_df)
    val_documents = build_agreement_documents(val_df)
    page_model_hyperparameters = _load_page_model_hyperparameters(optuna_path)
    _log(
        f"[review] using page-model hyperparameters: "
        + f"c1={page_model_hyperparameters['c1']:.6f}, "
        + f"c2={page_model_hyperparameters['c2']:.6f}, "
        + f"tfidf_max_features={page_model_hyperparameters['tfidf_max_features']}"
    )
    if not rebuild_feature_cache and train_oof_cache_path.exists() and val_cache_path.exists():
        _log(
            "[review] loading cached review features: "
            + f"train_oof={train_oof_cache_path}, val={val_cache_path}"
        )
        train_dataset = _load_cached_review_dataset(train_oof_cache_path)
        val_dataset = _load_cached_review_dataset(val_cache_path)
    else:
        if rebuild_feature_cache:
            _log("[review] rebuilding review feature cache from CRF predictions")
        else:
            _log("[review] cached review feature CSVs not found; building them now")
        train_dataset, val_dataset = _build_review_training_dataset(
            train_documents=train_documents,
            val_documents=val_documents,
            page_model_hyperparameters=page_model_hyperparameters,
            illegal_transition_weight=illegal_transition_weight,
            oof_splits=oof_splits,
            split_seed=split_seed,
        )
        _save_cached_review_dataset(train_dataset, train_oof_cache_path)
        _save_cached_review_dataset(val_dataset, val_cache_path)
        _log(
            "[review] wrote cached review features: "
            + f"train_oof={train_oof_cache_path}, val={val_cache_path}"
        )
    _log(
        f"[review] built datasets: train_oof_agreements={len(train_dataset.frame)}, "
        + f"val_agreements={len(val_dataset.frame)}, features={len(train_dataset.feature_columns)}"
    )
    _log("[review] selecting agreement-level review model")
    model_name, review_model, threshold_metrics, candidate_summaries = _select_best_review_model(
        train_dataset,
        val_dataset,
    )
    threshold = float(threshold_metrics["threshold"])
    _log(
        f"[review] selected model: {model_name}")
    _log(
        f"[review] selected threshold: threshold={threshold:.6f}, "
        + f"target_recall={float(threshold_metrics['target_recall']):.4f}, "
        + f"recall={float(threshold_metrics['recall']):.4f}, "
        + f"precision={float(threshold_metrics['precision']):.4f}, "
        + f"flagged_count={int(threshold_metrics['flagged_count'])}"
    )

    train_metrics = _evaluate_review_model(
        name="train_oof",
        dataset=train_dataset,
        model=review_model,
        threshold=threshold,
    )
    val_metrics = _evaluate_review_model(
        name="val",
        dataset=val_dataset,
        model=review_model,
        threshold=threshold,
    )

    artifact: ReviewArtifact = {
        "model": review_model,
        "model_name": model_name,
        "feature_columns": list(train_dataset.feature_columns),
        "threshold": threshold,
        "page_model_hyperparameters": page_model_hyperparameters,
        "summary_thresholds": {
            "low_confidence_threshold": DEFAULT_REVIEW_LOW_CONFIDENCE_THRESHOLD,
            "very_low_confidence_threshold": DEFAULT_REVIEW_VERY_LOW_CONFIDENCE_THRESHOLD,
            "low_margin_threshold": DEFAULT_REVIEW_LOW_MARGIN_THRESHOLD,
        },
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    _ = joblib.dump(artifact, model_path)

    model_feature_metric, model_feature_rows, model_intercept = _extract_model_feature_rows(
        review_model,
        train_dataset.feature_columns,
    )

    metrics_payload: dict[str, object] = {
        "split_meta": split_manifest["meta"],
        "feature_cache_paths": {
            "train_oof": str(train_oof_cache_path),
            "val": str(val_cache_path),
        },
        "page_model_hyperparameters": page_model_hyperparameters,
        "selected_model_name": model_name,
        "threshold_selection_dataset": "val",
        "selected_threshold": threshold_metrics,
        "candidate_models": candidate_summaries,
        "train_oof": train_metrics,
        "val": val_metrics,
        "feature_columns": train_dataset.feature_columns,
        "model_feature_metric": model_feature_metric,
        "model_feature_rows": model_feature_rows,
        "model_intercept": model_intercept,
    }
    _log(
        "[review] val summary: "
        + f"flagged={int(cast(int, val_metrics['flagged_agreement_count']))}, "
        + f"positives={int(cast(int, val_metrics['positive_agreement_count']))}"
    )
    _log(f"[review] writing metrics to {metrics_path}")
    _save_metrics(metrics_path, metrics_payload)
    _log(f"[review] saved artifact to {model_path}")
    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "selected_threshold": threshold,
        "val_flagged_agreement_count": cast(int, val_metrics["flagged_agreement_count"]),
    }


class ReviewModelInference:
    """Inference wrapper for the agreement-level review model."""

    def __init__(
        self,
        model_path: str = CLASSIFIER_REVIEW_MODEL_PATH,
        *,
        page_classifier: ClassifierInference | None = None,
    ):
        artifact = cast(ReviewArtifact, cast(object, joblib.load(model_path)))
        self.model = artifact["model"]
        self.feature_columns = list(artifact["feature_columns"])
        self.threshold = float(artifact["threshold"])
        self.page_classifier = page_classifier

    def predict_from_summaries(
        self,
        summaries: list[AgreementReviewSummary],
    ) -> list[ReviewPrediction]:
        feature_frame = review_summaries_to_frame(summaries)
        x_features = cast(pd.DataFrame, feature_frame[self.feature_columns])
        probabilities = cast(np.ndarray, self.model.predict_proba(x_features))[:, 1]
        outputs: list[ReviewPrediction] = []
        for row, probability in zip(feature_frame.to_dict("records"), probabilities.tolist(), strict=True):
            outputs.append(
                {
                    "agreement_uuid": str(row["agreement_uuid"]),
                    "needs_review": bool(probability >= self.threshold),
                    "review_probability": float(probability),
                    "review_threshold": self.threshold,
                    "review_score": float(row["review_score"]),
                }
            )
        return outputs

    def predict_from_dataframe(self, df: pd.DataFrame) -> list[ReviewPrediction]:
        if self.page_classifier is None:
            self.page_classifier = ClassifierInference()
        summaries = self.page_classifier.summarize_review_risk(df)
        return self.predict_from_summaries(summaries)


def main() -> None:
    args = parse_args()
    _ = train_review_model(
        data_path=args.data_path,
        split_path=args.split_path,
        optuna_path=args.optuna_path,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        train_oof_cache_path=args.train_oof_cache_path,
        val_cache_path=args.val_cache_path,
        rebuild_feature_cache=args.rebuild_feature_cache,
        n_splits=args.n_splits,
        fold_index=args.fold_index,
        val_fold_index=args.val_fold_index,
        split_seed=args.split_seed,
        oof_splits=args.oof_splits,
        illegal_transition_weight=args.illegal_transition_weight,
    )


if __name__ == "__main__":
    main()
