"""CRF-based page classifier revamp pipeline."""

from typing import TYPE_CHECKING

from .crf_pipeline import (
    PAGE_LABELS,
    append_next_page_annex_feature,
    build_feature_sequences,
    extract_page_features,
    fit_tfidf_vectorizer,
    train_and_evaluate,
)
from .inference import (
    AgreementReviewSummary,
    ClassifierInference,
    ClassifierPrediction,
    ClassifierProbs,
    build_agreement_review_summary,
    review_summaries_to_frame,
)
from .page_classifier_constants import CLASSIFIER_CRF_PATH, CLASSIFIER_LABEL_LIST

if TYPE_CHECKING:
    from .review_model import ReviewModelInference, ReviewPrediction, train_review_model

__all__ = [
    "PAGE_LABELS",
    "AgreementReviewSummary",
    "CLASSIFIER_CRF_PATH",
    "CLASSIFIER_LABEL_LIST",
    "ClassifierInference",
    "ClassifierPrediction",
    "ClassifierProbs",
    "ReviewModelInference",
    "ReviewPrediction",
    "append_next_page_annex_feature",
    "build_agreement_review_summary",
    "build_feature_sequences",
    "extract_page_features",
    "fit_tfidf_vectorizer",
    "review_summaries_to_frame",
    "train_review_model",
    "train_and_evaluate",
]


def __getattr__(name: str) -> object:
    if name in {"ReviewModelInference", "ReviewPrediction", "train_review_model"}:
        from . import review_model

        return getattr(review_model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
