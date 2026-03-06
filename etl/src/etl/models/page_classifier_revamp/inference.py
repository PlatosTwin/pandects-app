"""Inference wrapper for the CRF-based page classifier revamp."""
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypedDict, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .crf_pipeline import (
    AgreementDocument,
    FeatureDict,
    POSTPROCESS_ANNEX_ENTRY_MAX_RATIO,
    POSTPROCESS_BACK_WINDOW_SIZE,
    POSTPROCESS_BODY_LIKE_MIN_RATIO,
    POSTPROCESS_EARLY_BACK_RELATIVE_PAGE_MAX,
    POSTPROCESS_LATE_BACK_TRIGGER_RELATIVE_PAGE,
    PostprocessParameters,
    build_feature_sequences,
    postprocess_prediction_sequence,
)
from .page_classifier_constants import CLASSIFIER_CRF_PATH, CLASSIFIER_LABEL_LIST


class ClassifierProbs(TypedDict):
    front_matter: float
    toc: float
    body: float
    sig: float
    back_matter: float


class ClassifierPrediction(TypedDict):
    pred_class: str
    pred_probs: ClassifierProbs
    postprocess_modified: bool


class AgreementReviewSummary(TypedDict):
    agreement_uuid: str
    page_count: int
    front_matter_page_count: int
    toc_page_count: int
    body_page_count: int
    sig_page_count: int
    back_matter_page_count: int
    predicted_boundary_count: int
    body_back_matter_transition_count: int
    low_confidence_page_count: int
    very_low_confidence_page_count: int
    low_margin_page_count: int
    late_low_confidence_page_count: int
    boundary_low_confidence_count: int
    body_back_matter_competition_count: int
    late_body_back_matter_competition_count: int
    low_confidence_run_length_max: int
    very_low_confidence_run_length_max: int
    low_margin_run_length_max: int
    boundary_low_confidence_run_length_max: int
    body_back_matter_competition_run_length_max: int
    high_page_risk_count: int
    late_high_page_risk_count: int
    boundary_high_page_risk_count: int
    high_page_risk_run_length_max: int
    max_page_risk: float
    mean_page_risk: float
    top_page_risk_mean_3: float
    top_page_risk_mean_5: float
    min_top_probability: float
    mean_top_probability: float
    min_margin: float
    mean_margin: float
    lowest_top_probability_mean_3: float
    lowest_top_probability_mean_5: float
    lowest_margin_mean_3: float
    lowest_margin_mean_5: float
    review_score: float


DEFAULT_REVIEW_LOW_CONFIDENCE_THRESHOLD = 0.90
DEFAULT_REVIEW_VERY_LOW_CONFIDENCE_THRESHOLD = 0.75
DEFAULT_REVIEW_LOW_MARGIN_THRESHOLD = 0.20
DEFAULT_HIGH_PAGE_RISK_THRESHOLD = 3.0


class CRFModelProtocol(Protocol):
    def predict(self, x_test: list[list[FeatureDict]]) -> list[list[str]]: ...

    def predict_marginals(
        self,
        x_test: list[list[FeatureDict]],
    ) -> list[list[dict[str, float]]]: ...


@dataclass(frozen=True)
class _PreparedInferenceBatch:
    documents: list[AgreementDocument]
    row_indices_by_document: list[list[int]]
    row_count: int


def _default_postprocess_parameters() -> PostprocessParameters:
    return {
        "early_back_relative_page_max": POSTPROCESS_EARLY_BACK_RELATIVE_PAGE_MAX,
        "back_window_size": POSTPROCESS_BACK_WINDOW_SIZE,
        "body_like_min_ratio": POSTPROCESS_BODY_LIKE_MIN_RATIO,
        "annex_entry_max_ratio": POSTPROCESS_ANNEX_ENTRY_MAX_RATIO,
        "late_back_trigger_relative_page": POSTPROCESS_LATE_BACK_TRIGGER_RELATIVE_PAGE,
    }


def _coerce_postprocess_parameters(value: object) -> PostprocessParameters:
    if not isinstance(value, dict):
        return _default_postprocess_parameters()
    required_keys = {
        "early_back_relative_page_max",
        "back_window_size",
        "body_like_min_ratio",
        "annex_entry_max_ratio",
        "late_back_trigger_relative_page",
    }
    if required_keys - set(value):
        return _default_postprocess_parameters()
    return {
        "early_back_relative_page_max": float(value["early_back_relative_page_max"]),
        "back_window_size": int(value["back_window_size"]),
        "body_like_min_ratio": float(value["body_like_min_ratio"]),
        "annex_entry_max_ratio": float(value["annex_entry_max_ratio"]),
        "late_back_trigger_relative_page": float(value["late_back_trigger_relative_page"]),
    }


def build_agreement_review_summary(
    agreement_uuid: str,
    predicted_labels: list[str],
    marginal_rows: list[dict[str, float]],
    *,
    low_confidence_threshold: float = DEFAULT_REVIEW_LOW_CONFIDENCE_THRESHOLD,
    very_low_confidence_threshold: float = DEFAULT_REVIEW_VERY_LOW_CONFIDENCE_THRESHOLD,
    low_margin_threshold: float = DEFAULT_REVIEW_LOW_MARGIN_THRESHOLD,
    high_page_risk_threshold: float = DEFAULT_HIGH_PAGE_RISK_THRESHOLD,
) -> AgreementReviewSummary:
    front_matter_page_count = 0
    toc_page_count = 0
    body_page_count = 0
    sig_page_count = 0
    back_matter_page_count = 0
    top_probabilities: list[float] = []
    margins: list[float] = []
    predicted_boundary_count = 0
    body_back_matter_transition_count = 0
    low_confidence_page_count = 0
    very_low_confidence_page_count = 0
    low_margin_page_count = 0
    late_low_confidence_page_count = 0
    boundary_low_confidence_count = 0
    body_back_matter_competition_count = 0
    late_body_back_matter_competition_count = 0
    low_confidence_run_length_max = 0
    very_low_confidence_run_length_max = 0
    low_margin_run_length_max = 0
    boundary_low_confidence_run_length_max = 0
    body_back_matter_competition_run_length_max = 0
    high_page_risk_count = 0
    late_high_page_risk_count = 0
    boundary_high_page_risk_count = 0
    high_page_risk_run_length_max = 0
    low_confidence_run_length = 0
    very_low_confidence_run_length = 0
    low_margin_run_length = 0
    boundary_low_confidence_run_length = 0
    body_back_matter_competition_run_length = 0
    high_page_risk_run_length = 0
    late_page_start = max(len(predicted_labels) // 2, len(predicted_labels) - 10)
    page_risks: list[float] = []

    for page_index, (predicted_label, marginal_map) in enumerate(
        zip(predicted_labels, marginal_rows, strict=True)
    ):
        if predicted_label == "front_matter":
            front_matter_page_count += 1
        elif predicted_label == "toc":
            toc_page_count += 1
        elif predicted_label == "body":
            body_page_count += 1
        elif predicted_label == "sig":
            sig_page_count += 1
        elif predicted_label == "back_matter":
            back_matter_page_count += 1

        probs = sorted((float(prob) for prob in marginal_map.values()), reverse=True)
        top_probability = probs[0] if probs else 0.0
        second_probability = probs[1] if len(probs) > 1 else 0.0
        margin = top_probability - second_probability
        top_probabilities.append(top_probability)
        margins.append(margin)

        if top_probability < low_confidence_threshold:
            low_confidence_page_count += 1
            low_confidence_run_length += 1
            low_confidence_run_length_max = max(low_confidence_run_length_max, low_confidence_run_length)
            if page_index >= late_page_start:
                late_low_confidence_page_count += 1
        else:
            low_confidence_run_length = 0
        if top_probability < very_low_confidence_threshold:
            very_low_confidence_page_count += 1
            very_low_confidence_run_length += 1
            very_low_confidence_run_length_max = max(
                very_low_confidence_run_length_max,
                very_low_confidence_run_length,
            )
        else:
            very_low_confidence_run_length = 0
        if margin < low_margin_threshold:
            low_margin_page_count += 1
            low_margin_run_length += 1
            low_margin_run_length_max = max(low_margin_run_length_max, low_margin_run_length)
        else:
            low_margin_run_length = 0

        is_boundary_page = False
        if page_index > 0 and predicted_labels[page_index - 1] != predicted_label:
            is_boundary_page = True
            if {predicted_labels[page_index - 1], predicted_label} == {"body", "back_matter"}:
                body_back_matter_transition_count += 1
        if page_index + 1 < len(predicted_labels) and predicted_labels[page_index + 1] != predicted_label:
            is_boundary_page = True
        if is_boundary_page:
            predicted_boundary_count += 1
            if top_probability < low_confidence_threshold:
                boundary_low_confidence_count += 1
                boundary_low_confidence_run_length += 1
                boundary_low_confidence_run_length_max = max(
                    boundary_low_confidence_run_length_max,
                    boundary_low_confidence_run_length,
                )
            else:
                boundary_low_confidence_run_length = 0
        else:
            boundary_low_confidence_run_length = 0

        body_prob = float(marginal_map.get("body", 0.0))
        back_prob = float(marginal_map.get("back_matter", 0.0))
        body_back_matter_competition = (
            predicted_label in {"body", "back_matter"} and min(body_prob, back_prob) >= 0.05
        )
        if body_back_matter_competition:
            body_back_matter_competition_count += 1
            body_back_matter_competition_run_length += 1
            body_back_matter_competition_run_length_max = max(
                body_back_matter_competition_run_length_max,
                body_back_matter_competition_run_length,
            )
            if page_index >= late_page_start:
                late_body_back_matter_competition_count += 1
        else:
            body_back_matter_competition_run_length = 0

        page_risk = (
            (1.0 - top_probability) * 3.0
            + (1.0 - margin) * 2.0
            + (1.25 if top_probability < very_low_confidence_threshold else 0.0)
            + (0.75 if top_probability < low_confidence_threshold else 0.0)
            + (0.75 if margin < low_margin_threshold else 0.0)
            + (0.75 if is_boundary_page else 0.0)
            + (1.0 if body_back_matter_competition else 0.0)
        )
        page_risks.append(page_risk)
        if page_risk >= high_page_risk_threshold:
            high_page_risk_count += 1
            high_page_risk_run_length += 1
            high_page_risk_run_length_max = max(high_page_risk_run_length_max, high_page_risk_run_length)
            if is_boundary_page:
                boundary_high_page_risk_count += 1
            if page_index >= late_page_start:
                late_high_page_risk_count += 1
        else:
            high_page_risk_run_length = 0

    page_count = len(predicted_labels)
    mean_top_probability = float(sum(top_probabilities)) / float(page_count) if page_count > 0 else 0.0
    mean_margin = float(sum(margins)) / float(page_count) if page_count > 0 else 0.0
    min_top_probability = min(top_probabilities) if top_probabilities else 0.0
    min_margin = min(margins) if margins else 0.0
    lowest_top_probabilities = sorted(top_probabilities)
    lowest_margins = sorted(margins)
    lowest_top_probability_mean_3 = (
        float(sum(lowest_top_probabilities[:3])) / float(min(3, page_count))
        if page_count > 0
        else 0.0
    )
    lowest_top_probability_mean_5 = (
        float(sum(lowest_top_probabilities[:5])) / float(min(5, page_count))
        if page_count > 0
        else 0.0
    )
    highest_page_risks = sorted(page_risks, reverse=True)
    max_page_risk = highest_page_risks[0] if highest_page_risks else 0.0
    mean_page_risk = float(sum(page_risks)) / float(page_count) if page_count > 0 else 0.0
    top_page_risk_mean_3 = (
        float(sum(highest_page_risks[:3])) / float(min(3, page_count))
        if page_count > 0
        else 0.0
    )
    top_page_risk_mean_5 = (
        float(sum(highest_page_risks[:5])) / float(min(5, page_count))
        if page_count > 0
        else 0.0
    )
    lowest_margin_mean_3 = (
        float(sum(lowest_margins[:3])) / float(min(3, page_count))
        if page_count > 0
        else 0.0
    )
    lowest_margin_mean_5 = (
        float(sum(lowest_margins[:5])) / float(min(5, page_count))
        if page_count > 0
        else 0.0
    )
    review_score = (
        float(very_low_confidence_page_count) * 3.0
        + float(boundary_low_confidence_count) * 2.0
        + float(low_margin_page_count)
        + float(body_back_matter_competition_count) * 0.5
        + float(low_confidence_run_length_max) * 1.5
        + float(low_margin_run_length_max) * 1.5
        + float(body_back_matter_competition_run_length_max)
        + float(high_page_risk_run_length_max) * 1.5
        + max_page_risk
        + (1.0 - min_top_probability) * 10.0
        + (1.0 - min_margin) * 5.0
    )
    return {
        "agreement_uuid": agreement_uuid,
        "page_count": page_count,
        "front_matter_page_count": front_matter_page_count,
        "toc_page_count": toc_page_count,
        "body_page_count": body_page_count,
        "sig_page_count": sig_page_count,
        "back_matter_page_count": back_matter_page_count,
        "predicted_boundary_count": predicted_boundary_count,
        "body_back_matter_transition_count": body_back_matter_transition_count,
        "low_confidence_page_count": low_confidence_page_count,
        "very_low_confidence_page_count": very_low_confidence_page_count,
        "low_margin_page_count": low_margin_page_count,
        "late_low_confidence_page_count": late_low_confidence_page_count,
        "boundary_low_confidence_count": boundary_low_confidence_count,
        "body_back_matter_competition_count": body_back_matter_competition_count,
        "late_body_back_matter_competition_count": late_body_back_matter_competition_count,
        "low_confidence_run_length_max": low_confidence_run_length_max,
        "very_low_confidence_run_length_max": very_low_confidence_run_length_max,
        "low_margin_run_length_max": low_margin_run_length_max,
        "boundary_low_confidence_run_length_max": boundary_low_confidence_run_length_max,
        "body_back_matter_competition_run_length_max": body_back_matter_competition_run_length_max,
        "high_page_risk_count": high_page_risk_count,
        "late_high_page_risk_count": late_high_page_risk_count,
        "boundary_high_page_risk_count": boundary_high_page_risk_count,
        "high_page_risk_run_length_max": high_page_risk_run_length_max,
        "max_page_risk": max_page_risk,
        "mean_page_risk": mean_page_risk,
        "top_page_risk_mean_3": top_page_risk_mean_3,
        "top_page_risk_mean_5": top_page_risk_mean_5,
        "min_top_probability": min_top_probability,
        "mean_top_probability": mean_top_probability,
        "min_margin": min_margin,
        "mean_margin": mean_margin,
        "lowest_top_probability_mean_3": lowest_top_probability_mean_3,
        "lowest_top_probability_mean_5": lowest_top_probability_mean_5,
        "lowest_margin_mean_3": lowest_margin_mean_3,
        "lowest_margin_mean_5": lowest_margin_mean_5,
        "review_score": review_score,
    }


def review_summaries_to_frame(
    summaries: list[AgreementReviewSummary],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for summary in summaries:
        page_count = max(int(summary["page_count"]), 1)
        predicted_boundary_count = int(summary["predicted_boundary_count"])
        low_confidence_page_count = int(summary["low_confidence_page_count"])
        very_low_confidence_page_count = int(summary["very_low_confidence_page_count"])
        low_margin_page_count = int(summary["low_margin_page_count"])
        boundary_low_confidence_count = int(summary["boundary_low_confidence_count"])
        body_back_matter_competition_count = int(summary["body_back_matter_competition_count"])
        rows.append(
            {
                "agreement_uuid": str(summary["agreement_uuid"]),
                "page_count": float(page_count),
                "front_matter_page_count": float(summary["front_matter_page_count"]),
                "front_matter_page_ratio": float(summary["front_matter_page_count"]) / float(page_count),
                "toc_page_count": float(summary["toc_page_count"]),
                "toc_page_ratio": float(summary["toc_page_count"]) / float(page_count),
                "body_page_count": float(summary["body_page_count"]),
                "body_page_ratio": float(summary["body_page_count"]) / float(page_count),
                "sig_page_count": float(summary["sig_page_count"]),
                "sig_page_ratio": float(summary["sig_page_count"]) / float(page_count),
                "back_matter_page_count": float(summary["back_matter_page_count"]),
                "back_matter_page_ratio": float(summary["back_matter_page_count"]) / float(page_count),
                "predicted_boundary_count": float(predicted_boundary_count),
                "predicted_boundary_ratio": float(predicted_boundary_count) / float(page_count),
                "body_back_matter_transition_count": float(summary["body_back_matter_transition_count"]),
                "body_back_matter_transition_ratio": float(summary["body_back_matter_transition_count"])
                / float(page_count),
                "low_confidence_page_count": float(low_confidence_page_count),
                "low_confidence_page_ratio": float(low_confidence_page_count) / float(page_count),
                "very_low_confidence_page_count": float(very_low_confidence_page_count),
                "very_low_confidence_page_ratio": float(very_low_confidence_page_count) / float(page_count),
                "low_margin_page_count": float(low_margin_page_count),
                "low_margin_page_ratio": float(low_margin_page_count) / float(page_count),
                "late_low_confidence_page_count": float(summary["late_low_confidence_page_count"]),
                "late_low_confidence_page_ratio": float(summary["late_low_confidence_page_count"])
                / float(page_count),
                "boundary_low_confidence_count": float(boundary_low_confidence_count),
                "boundary_low_confidence_ratio": float(boundary_low_confidence_count) / float(page_count),
                "body_back_matter_competition_count": float(body_back_matter_competition_count),
                "body_back_matter_competition_ratio": float(body_back_matter_competition_count) / float(page_count),
                "late_body_back_matter_competition_count": float(
                    summary["late_body_back_matter_competition_count"]
                ),
                "late_body_back_matter_competition_ratio": float(
                    summary["late_body_back_matter_competition_count"]
                )
                / float(page_count),
                "low_confidence_run_length_max": float(summary["low_confidence_run_length_max"]),
                "very_low_confidence_run_length_max": float(
                    summary["very_low_confidence_run_length_max"]
                ),
                "low_margin_run_length_max": float(summary["low_margin_run_length_max"]),
                "boundary_low_confidence_run_length_max": float(
                    summary["boundary_low_confidence_run_length_max"]
                ),
                "body_back_matter_competition_run_length_max": float(
                    summary["body_back_matter_competition_run_length_max"]
                ),
                "high_page_risk_count": float(summary["high_page_risk_count"]),
                "high_page_risk_ratio": float(summary["high_page_risk_count"]) / float(page_count),
                "late_high_page_risk_count": float(summary["late_high_page_risk_count"]),
                "late_high_page_risk_ratio": float(summary["late_high_page_risk_count"])
                / float(page_count),
                "boundary_high_page_risk_count": float(summary["boundary_high_page_risk_count"]),
                "boundary_high_page_risk_ratio": float(summary["boundary_high_page_risk_count"])
                / float(page_count),
                "high_page_risk_run_length_max": float(summary["high_page_risk_run_length_max"]),
                "max_page_risk": float(summary["max_page_risk"]),
                "mean_page_risk": float(summary["mean_page_risk"]),
                "top_page_risk_mean_3": float(summary["top_page_risk_mean_3"]),
                "top_page_risk_mean_5": float(summary["top_page_risk_mean_5"]),
                "min_top_probability": float(summary["min_top_probability"]),
                "mean_top_probability": float(summary["mean_top_probability"]),
                "min_margin": float(summary["min_margin"]),
                "mean_margin": float(summary["mean_margin"]),
                "lowest_top_probability_mean_3": float(summary["lowest_top_probability_mean_3"]),
                "lowest_top_probability_mean_5": float(summary["lowest_top_probability_mean_5"]),
                "lowest_margin_mean_3": float(summary["lowest_margin_mean_3"]),
                "lowest_margin_mean_5": float(summary["lowest_margin_mean_5"]),
                "review_score": float(summary["review_score"]),
            }
        )
    return pd.DataFrame(rows)


class ClassifierInference:
    """Wrapper around the trained CRF artifact for batch inference."""

    def __init__(self, model_path: str = CLASSIFIER_CRF_PATH):
        artifact = cast(dict[str, object], joblib.load(model_path))
        self.labels = [str(label) for label in cast(list[object], artifact["labels"])]
        if self.labels != CLASSIFIER_LABEL_LIST:
            raise ValueError(
                "Loaded CRF artifact label list does not match the production page label list."
            )
        self.model = cast(CRFModelProtocol, artifact["crf_model"])
        self.vectorizer = cast(TfidfVectorizer, artifact["vectorizer"])
        self.postprocess_parameters = _coerce_postprocess_parameters(
            artifact.get("postprocess_parameters")
        )

    def classify(self, df: pd.DataFrame) -> list[ClassifierPrediction]:
        prepared = self._prepare_dataframe(df)
        predictions, marginals, postprocess_modified_masks = self._predict_with_marginals(
            prepared.documents
        )
        output_by_row_index: dict[int, ClassifierPrediction] = {}
        for row_indices, predicted_labels, marginal_rows, modified_mask in zip(
            prepared.row_indices_by_document,
            predictions,
            marginals,
            postprocess_modified_masks,
            strict=True,
        ):
            for row_index, predicted_label, marginal_map, postprocess_modified in zip(
                row_indices,
                predicted_labels,
                marginal_rows,
                modified_mask,
                strict=True,
            ):
                pred_probs: ClassifierProbs = {
                    "front_matter": float(marginal_map.get("front_matter", 0.0)),
                    "toc": float(marginal_map.get("toc", 0.0)),
                    "body": float(marginal_map.get("body", 0.0)),
                    "sig": float(marginal_map.get("sig", 0.0)),
                    "back_matter": float(marginal_map.get("back_matter", 0.0)),
                }
                output_by_row_index[row_index] = {
                    "pred_class": str(predicted_label),
                    "pred_probs": pred_probs,
                    "postprocess_modified": bool(postprocess_modified),
                }
        return [output_by_row_index[row_index] for row_index in range(prepared.row_count)]

    def summarize_review_risk(
        self,
        df: pd.DataFrame,
        *,
        low_confidence_threshold: float = DEFAULT_REVIEW_LOW_CONFIDENCE_THRESHOLD,
        very_low_confidence_threshold: float = DEFAULT_REVIEW_VERY_LOW_CONFIDENCE_THRESHOLD,
        low_margin_threshold: float = DEFAULT_REVIEW_LOW_MARGIN_THRESHOLD,
    ) -> list[AgreementReviewSummary]:
        prepared = self._prepare_dataframe(df)
        predictions, marginals, _ = self._predict_with_marginals(prepared.documents)
        summaries = [
            build_agreement_review_summary(
                document.agreement_uuid,
                predicted_labels,
                marginal_rows,
                low_confidence_threshold=low_confidence_threshold,
                very_low_confidence_threshold=very_low_confidence_threshold,
                low_margin_threshold=low_margin_threshold,
            )
            for document, predicted_labels, marginal_rows in zip(
            prepared.documents,
            predictions,
            marginals,
            strict=True,
            )
        ]

        return sorted(
            summaries,
            key=lambda summary: (
                -summary["review_score"],
                summary["agreement_uuid"],
            ),
        )

    def _prepare_dataframe(self, df: pd.DataFrame) -> _PreparedInferenceBatch:
        required_columns = {"agreement_uuid", "text", "order"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"Inference DataFrame must contain columns {sorted(required_columns)}. "
                + f"Missing: {sorted(missing_columns)}"
            )

        working = df.copy()
        working["_row_index"] = np.arange(len(working), dtype=np.int64)
        working["agreement_uuid"] = working["agreement_uuid"].astype(str)
        working["text"] = working["text"].fillna("").astype(str)
        working["order"] = np.asarray(
            pd.to_numeric(working["order"], errors="raise"),
            dtype=np.int64,
        )
        if bool((working["order"] < 0).any()):
            raise ValueError("Inference `order` values must be non-negative.")

        sorted_working = working.sort_values(
            ["agreement_uuid", "order", "_row_index"],
            kind="mergesort",
        ).reset_index(drop=True)

        documents: list[AgreementDocument] = []
        row_indices_by_document: list[list[int]] = []
        for agreement_uuid, group in sorted_working.groupby("agreement_uuid", sort=False):
            documents.append(
                AgreementDocument(
                    agreement_uuid=str(agreement_uuid),
                    page_texts=[str(text) for text in group["text"].tolist()],
                    labels=[],
                )
            )
            row_indices_by_document.append([int(row_index) for row_index in group["_row_index"].tolist()])

        return _PreparedInferenceBatch(
            documents=documents,
            row_indices_by_document=row_indices_by_document,
            row_count=len(working),
        )

    def _predict_with_marginals(
        self,
        documents: list[AgreementDocument],
    ) -> tuple[list[list[str]], list[list[dict[str, float]]], list[list[bool]]]:
        sequences = build_feature_sequences(documents, self.vectorizer)
        predictions_raw = self.model.predict(sequences)
        marginals_raw = self.model.predict_marginals(sequences)
        predictions = [[str(label) for label in sequence] for sequence in predictions_raw]
        marginals = [
            [{str(label): float(prob) for label, prob in marginal.items()} for marginal in sequence]
            for sequence in marginals_raw
        ]
        postprocess_modified_masks: list[list[bool]] = []
        processed_predictions: list[list[str]] = []
        for predicted_labels, feature_sequence in zip(predictions, sequences, strict=True):
            processed_labels, modified_mask = postprocess_prediction_sequence(
                predicted_labels,
                feature_sequence,
                postprocess_parameters=self.postprocess_parameters,
            )
            processed_predictions.append(processed_labels)
            postprocess_modified_masks.append(modified_mask)
        return processed_predictions, marginals, postprocess_modified_masks
