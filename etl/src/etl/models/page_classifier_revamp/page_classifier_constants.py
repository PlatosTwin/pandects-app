"""Constants for the CRF-based page classifier revamp."""

from pathlib import Path

from .crf_pipeline import PAGE_LABELS

MODEL_FILES_DIR = Path(__file__).parent / "model_files"
EVAL_METRICS_DIR = Path(__file__).parent / "eval_metrics"

CLASSIFIER_LABEL_LIST = list(PAGE_LABELS)
CLASSIFIER_CRF_PATH = str(MODEL_FILES_DIR / "page-classifier-crf_final_test.joblib")
CLASSIFIER_CRF_TUNE_VAL_PATH = str(MODEL_FILES_DIR / "page-classifier-crf_tune_val.joblib")
CLASSIFIER_REVIEW_MODEL_PATH = str(MODEL_FILES_DIR / "page-classifier-review-model_dev_val.joblib")
CLASSIFIER_REVIEW_METRICS_PATH = str(EVAL_METRICS_DIR / "page_classifier_review_model_metrics_dev_val.json")
