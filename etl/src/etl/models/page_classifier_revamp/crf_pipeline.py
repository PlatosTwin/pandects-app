"""CRF pipeline for monotonic M&A agreement page classification."""
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, Protocol, cast

import joblib
import numpy as np
import optuna
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

PAGE_LABELS = ["front_matter", "toc", "body", "sig", "back_matter"]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(PAGE_LABELS)}
AGREEMENT_TYPE_LABELS = [
    "no_sig",
    "no_back_matter",
    "sig_plus_normal_back_matter",
    "sig_plus_massive_back_matter",
]
NORMAL_BACK_MATTER_MAX_PAGES = 30
REQUIRED_COLUMNS = {
    "page_uuid",
    "agreement_uuid",
    "raw_page_content",
    "text",
    "order",
    "label",
    "date_announcement",
}

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "page-data.parquet"
DEFAULT_SPLIT_PATH = BASE_DIR / "data" / "agreement-splits.json"
DEFAULT_MODEL_PATH = BASE_DIR / "model_files" / "page-classifier-crf.joblib"
DEFAULT_REPORT_PATH = BASE_DIR / "eval_metrics" / "page_classifier_revamp_classification_report.txt"
DEFAULT_METRICS_PATH = BASE_DIR / "eval_metrics" / "page_classifier_revamp_metrics.json"
DEFAULT_SPLIT_SEED = 2718
DEFAULT_OPTUNA_PATH = BASE_DIR / "eval_metrics" / "page_classifier_revamp_optuna_best.json"
DEFAULT_OPTUNA_TRIALS = 12

TOC_DOTS_RE = re.compile(r"\.{6,}")
SIG_BLOCK_RE = re.compile(
    r"\bIN WITNESS WHEREOF\b|^\s*(?:By|Title)\s*:",
    re.IGNORECASE | re.MULTILINE,
)
ANNEX_ANCHOR_RE = re.compile(
    r"^\s*(?:EXHIBIT|SCHEDULE|ANNEX|APPENDIX|ATTACHMENT)\b(?:\s+[A-Z0-9][A-Z0-9 .()/_-]*)?\s*$"
)
APPENDIX_ANCHOR_RE = re.compile(
    r"^\s*(?:APPENDIX|ATTACHMENT)\b(?:\s+[A-Z0-9][A-Z0-9 .()/_-]*)?\s*$"
)
DEFINITIONS_HEADING_RE = re.compile(r"^\s*definitions?\s*$", re.IGNORECASE)
LIST_OF_EXHIBITS_RE = re.compile(r"\blist of exhibits\b", re.IGNORECASE)
LIST_OF_SCHEDULES_RE = re.compile(r"\blist of schedules\b", re.IGNORECASE)
PARTICULARS_RE = re.compile(r"\bparticulars of\b", re.IGNORECASE)
FORM_OF_RE = re.compile(r"\bform of\b", re.IGNORECASE)
ARTICLE_RE = re.compile(r"\barticle\b", re.IGNORECASE)
SECTION_RE = re.compile(r"\bsection\b", re.IGNORECASE)
ARTICLE_HEADING_RE = re.compile(r"^\s*article\s+[0-9ivxlcdm]+\b", re.IGNORECASE)
SECTION_HEADING_RE = re.compile(
    r"^\s*(?:section|clause)?\s*\d+(?:\.\d+)*(?:\([a-z0-9]+\))*[\.\)]?\s+[A-Z]",
    re.IGNORECASE,
)
ANNEX_KEYWORD_RE = re.compile(
    r"\b(?:schedule|schedules|exhibit|exhibits|annex|annexes|appendix|appendices|attachment|attachments)\b",
    re.IGNORECASE,
)
DEFINED_TERM_RE = re.compile(r"[\"“][A-Z][^\"”]{1,80}[\"”]")
EXECUTED_AS_DEED_RE = re.compile(r"\bexecuted\s+\(?.{0,30}as a deed\b", re.IGNORECASE)
WITNESS_SIGNATURE_RE = re.compile(r"\bsignature of witness\b", re.IGNORECASE)
WITNESS_ADDRESS_RE = re.compile(r"\bwitness name\s*:|\baddress\s*:", re.IGNORECASE)
WITNESS_OCCUPATION_RE = re.compile(r"\boccupation\s*:", re.IGNORECASE)
COUNTERPART_RE = re.compile(r"\bcounterparts?\b", re.IGNORECASE)
TABLE_OF_CONTENTS_RE = re.compile(r"^\s*table of contents\b", re.IGNORECASE)
INTENTIONALLY_OMITTED_RE = re.compile(r"\bintentionally omitted\b", re.IGNORECASE)
STANDALONE_AGREEMENT_TITLE_RE = re.compile(
    r"^\s*(?:form of\s+)?[A-Z][A-Z0-9 ,.'&()/:-]{0,120}\bagreement\b",
    re.IGNORECASE,
)
LEADING_FRAGMENT_RE = re.compile(r"[A-Z][A-Za-z&'/-]*(?:\s+[A-Z][A-Za-z&'/-]*){0,5}")

FeatureValue = float | bool
FeatureDict = dict[str, FeatureValue]


@dataclass(frozen=True)
class AgreementDocument:
    agreement_uuid: str
    page_texts: list[str]
    labels: list[str]


@dataclass(frozen=True)
class LabelViolation:
    agreement_uuid: str
    previous_label: str
    current_label: str
    previous_order: int
    current_order: int


class SplitMeta(TypedDict):
    n_splits: int
    test_fold_index: int
    val_fold_index: int
    split_seed: int
    stratification_strategy: str


class SplitManifest(TypedDict):
    train: list[str]
    val: list[str]
    test: list[str]
    meta: SplitMeta
    details: dict[str, object]


class CRFHyperparameters(TypedDict):
    c1: float
    c2: float
    tfidf_max_features: int


def _log(message: str) -> None:
    print(message, flush=True)


def _coerce_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Split manifest field `{field_name}` must be an integer.")
    return value


def _coerce_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Split manifest field `{field_name}` must be a string.")
    return value


def _coerce_dict(value: object, *, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"Split manifest field `{field_name}` must be a mapping.")
    return {str(key): item for key, item in value.items()}


class CRFModelProtocol(Protocol):
    transition_features_: dict[tuple[str, str], float]

    def fit(
        self,
        x_train: list[list[FeatureDict]],
        y_train: list[list[str]],
    ) -> object: ...

    def predict(self, x_test: list[list[FeatureDict]]) -> list[list[str]]: ...


class CRFSuiteModuleProtocol(Protocol):
    def CRF(
        self,
        *,
        algorithm: str,
        c1: float,
        c2: float,
        max_iterations: int,
        all_possible_transitions: bool,
    ) -> CRFModelProtocol: ...


class CRFMetricsModuleProtocol(Protocol):
    def flat_classification_report(
        self,
        y_true: list[list[str]],
        y_pred: list[list[str]],
        *,
        labels: list[str],
        digits: int,
    ) -> str: ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a monotonic CRF page classifier."
    )
    _ = parser.add_argument(
        "--mode",
        choices=("train", "tune"),
        default="train",
        help="`train` runs one fit/eval pass; `tune` runs a small Optuna search on train/val.",
    )
    _ = parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Parquet dataset path.",
    )
    _ = parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to save the trained CRF artifact.",
    )
    _ = parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Where to save the flat classification report.",
    )
    _ = parser.add_argument(
        "--metrics-path",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Where to save the structured evaluation metrics JSON.",
    )
    _ = parser.add_argument(
        "--optuna-path",
        type=Path,
        default=DEFAULT_OPTUNA_PATH,
        help="Where to save the best Optuna trial summary.",
    )
    _ = parser.add_argument(
        "--split-path",
        type=Path,
        default=DEFAULT_SPLIT_PATH,
        help="Stable train/val/test agreement split manifest.",
    )
    _ = parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of GroupKFold folds used to build the manifest.",
    )
    _ = parser.add_argument(
        "--fold-index",
        type=int,
        default=0,
        help="Zero-based GroupKFold test fold index for manifest creation.",
    )
    _ = parser.add_argument(
        "--val-fold-index",
        type=int,
        default=None,
        help="Optional val fold index. Defaults to the next fold after the test fold.",
    )
    _ = parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Seed used when assigning agreements across train/val/test within strata.",
    )
    _ = parser.add_argument(
        "--fit-split",
        choices=("train", "train_val"),
        default="train",
        help="Use `train` during development. Use `train_val` only for a final model after tuning.",
    )
    _ = parser.add_argument(
        "--eval-split",
        choices=("val", "test"),
        default="val",
        help="Development should evaluate on `val`. Reserve `test` for final evaluation only.",
    )
    _ = parser.add_argument(
        "--optuna-trials",
        type=int,
        default=DEFAULT_OPTUNA_TRIALS,
        help="Number of Optuna trials when running in `tune` mode.",
    )
    _ = parser.add_argument(
        "--use-best-optuna-params",
        action="store_true",
        help="In `train` mode, load c1/c2/tfidf_max_features from the Optuna summary file.",
    )
    _ = parser.add_argument(
        "--illegal-transition-weight",
        type=float,
        default=-1000.0,
        help="Weight assigned to forbidden backward transitions.",
    )
    return parser.parse_args()


def _load_crf_modules() -> tuple[CRFSuiteModuleProtocol, CRFMetricsModuleProtocol]:
    try:
        sklearn_crfsuite = cast(
            CRFSuiteModuleProtocol,
            cast(object, importlib.import_module("sklearn_crfsuite")),
        )
        crf_metrics = cast(
            CRFMetricsModuleProtocol,
            cast(object, importlib.import_module("sklearn_crfsuite.metrics")),
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing CRF dependency: install `sklearn-crfsuite` in the ETL environment "
            + "before training this pipeline."
        ) from exc
    return sklearn_crfsuite, crf_metrics


def _coerce_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _normalize_page_text(text_value: object, raw_value: object) -> str:
    text = _coerce_text(text_value).strip()
    if text:
        return text
    return _coerce_text(raw_value).strip()


def load_page_dataframe(data_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(data_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset must contain columns {sorted(REQUIRED_COLUMNS)}. Missing: {sorted(missing)}"
        )

    working = df.copy()
    working["agreement_uuid"] = working["agreement_uuid"].astype(str)
    working["page_uuid"] = working["page_uuid"].astype(str)
    working["label"] = working["label"].astype(str)
    order_array = np.asarray(pd.to_numeric(working["order"], errors="raise"), dtype=np.int64)
    working["order"] = order_array.tolist()
    if (working["order"] < 0).any():
        raise ValueError("Page `order` values must be non-negative.")

    unknown_labels = sorted(set(working["label"]) - set(PAGE_LABELS))
    if unknown_labels:
        raise ValueError(f"Unexpected labels found: {unknown_labels}")

    working["page_text"] = [
        _normalize_page_text(text_value, raw_value)
        for text_value, raw_value in zip(
            working["text"],
            working["raw_page_content"],
            strict=True,
        )
    ]
    working = working.sort_values(["agreement_uuid", "order", "page_uuid"]).reset_index(
        drop=True
    )
    return working


def find_label_monotonicity_violations(df: pd.DataFrame) -> list[LabelViolation]:
    violations: list[LabelViolation] = []
    for agreement_uuid, group in df.groupby("agreement_uuid", sort=False):
        labels = group["label"].tolist()
        orders = group["order"].tolist()
        for index in range(1, len(labels)):
            previous_label = str(labels[index - 1])
            current_label = str(labels[index])
            if LABEL_TO_INDEX[current_label] < LABEL_TO_INDEX[previous_label]:
                violations.append(
                    LabelViolation(
                        agreement_uuid=str(agreement_uuid),
                        previous_label=previous_label,
                        current_label=current_label,
                        previous_order=int(orders[index - 1]),
                        current_order=int(orders[index]),
                    )
                )
                break
    return violations


def _count_leading_heading_fragments(page_text: str) -> int:
    leading_text = page_text[:220].replace("\n", " ")
    matches = [
        match.group().strip()
        for match in LEADING_FRAGMENT_RE.finditer(leading_text)
        if len(match.group().split()) <= 6 and len(match.group().strip()) >= 4
    ]
    fragment_count = 0
    for fragment in matches:
        alpha_tokens = re.findall(r"[A-Za-z][A-Za-z&'/-]*", fragment)
        if len(alpha_tokens) < 2:
            continue
        fragment_count += 1
    return fragment_count


def extract_page_features(page_text: str, page_number: int, total_pages: int) -> FeatureDict:
    safe_total_pages = max(total_pages, 1)
    char_count = max(len(page_text), 1)
    lines = page_text.splitlines() or [page_text]
    stripped_lines = [line.strip() for line in lines]
    avg_line_length = float(sum(len(line) for line in stripped_lines)) / max(len(stripped_lines), 1)
    short_line_ratio = float(sum(len(line) <= 40 for line in stripped_lines)) / max(len(stripped_lines), 1)

    alpha_tokens = re.findall(r"[A-Za-z][A-Za-z&'.-]*", page_text)
    all_caps_tokens = [
        token
        for token in alpha_tokens
        if len(token) > 1 and token.upper() == token
    ]
    all_caps_ratio = float(len(all_caps_tokens)) / max(len(alpha_tokens), 1)

    top_lines = [line for line in stripped_lines if line][:5]
    has_annex_anchor = any(ANNEX_ANCHOR_RE.match(line.upper()) for line in top_lines)
    has_appendix_anchor = any(APPENDIX_ANCHOR_RE.match(line.upper()) for line in top_lines)
    has_definitions_heading = any(DEFINITIONS_HEADING_RE.match(line) for line in top_lines)
    has_article_heading = any(ARTICLE_HEADING_RE.match(line) for line in top_lines)
    has_section_heading = any(SECTION_HEADING_RE.match(line) for line in top_lines)
    starts_with_table_of_contents = any(TABLE_OF_CONTENTS_RE.match(line) for line in top_lines[:2])
    has_standalone_agreement_title = any(
        STANDALONE_AGREEMENT_TITLE_RE.match(line) for line in top_lines[:3]
    )
    has_intentionally_omitted = bool(INTENTIONALLY_OMITTED_RE.search(" ".join(top_lines[:3])))
    non_empty_lines = [line for line in stripped_lines if line]
    heading_like_lines = [
        line
        for line in non_empty_lines[:8]
        if any(char.isalpha() for char in line) and line == line.upper()
    ]
    heading_line_ratio = float(len(heading_like_lines)) / max(len(non_empty_lines[:8]), 1)
    quoted_term_count = len(DEFINED_TERM_RE.findall(page_text))
    annex_keyword_count = len(ANNEX_KEYWORD_RE.findall(page_text))
    section_heading_count = sum(bool(SECTION_HEADING_RE.match(line)) for line in non_empty_lines[:12])
    article_heading_count = sum(bool(ARTICLE_HEADING_RE.match(line)) for line in non_empty_lines[:12])
    leading_heading_fragment_count = _count_leading_heading_fragments(page_text)
    page_index = page_number - 1
    pages_remaining = max(total_pages - page_number, 0)
    has_executed_as_deed = bool(EXECUTED_AS_DEED_RE.search(page_text))
    has_witness_signature = bool(WITNESS_SIGNATURE_RE.search(page_text))
    has_witness_address = bool(WITNESS_ADDRESS_RE.search(page_text))
    has_witness_occupation = bool(WITNESS_OCCUPATION_RE.search(page_text))
    witness_block_combo = (
        has_executed_as_deed and (has_witness_signature or has_witness_address or has_witness_occupation)
    )

    return {
        "page_index": float(page_index),
        "pages_remaining": float(pages_remaining),
        "relative_page": float(page_number) / float(safe_total_pages),
        "in_first_quintile": page_number <= math.ceil(safe_total_pages * 0.2),
        "in_first_decile": page_number <= math.ceil(safe_total_pages * 0.1),
        "in_last_quintile": page_number > (safe_total_pages - math.ceil(safe_total_pages * 0.2)),
        "in_last_decile": page_number > (safe_total_pages - math.ceil(safe_total_pages * 0.1)),
        "whitespace_ratio": float(sum(char.isspace() for char in page_text)) / float(char_count),
        "avg_line_length": avg_line_length,
        "short_line_ratio": short_line_ratio,
        "heading_line_ratio": heading_line_ratio,
        "is_all_caps_heavy": all_caps_ratio >= 0.35,
        "has_article_heading": has_article_heading,
        "has_section_heading": has_section_heading,
        "article_heading_count": float(article_heading_count),
        "section_heading_count": float(section_heading_count),
        "section_mentions": float(len(SECTION_RE.findall(page_text))),
        "article_mentions": float(len(ARTICLE_RE.findall(page_text))),
        "has_toc_dots": bool(TOC_DOTS_RE.search(page_text)),
        "has_sig_block": bool(SIG_BLOCK_RE.search(page_text)),
        "has_annex_anchor": has_annex_anchor,
        "has_appendix_anchor": has_appendix_anchor,
        "has_definitions_heading": has_definitions_heading,
        "starts_with_table_of_contents": starts_with_table_of_contents,
        "has_standalone_agreement_title": has_standalone_agreement_title,
        "has_intentionally_omitted": has_intentionally_omitted,
        "has_list_of_exhibits": bool(LIST_OF_EXHIBITS_RE.search(page_text)),
        "has_list_of_schedules": bool(LIST_OF_SCHEDULES_RE.search(page_text)),
        "has_particulars_heading": bool(PARTICULARS_RE.search(page_text)),
        "has_form_of": bool(FORM_OF_RE.search(page_text)),
        "has_counterparts": bool(COUNTERPART_RE.search(page_text)),
        "has_executed_as_deed": has_executed_as_deed,
        "has_witness_signature": has_witness_signature,
        "has_witness_address": has_witness_address,
        "has_witness_occupation": has_witness_occupation,
        "witness_block_combo": witness_block_combo,
        "annex_keyword_count": float(annex_keyword_count),
        "leading_heading_fragment_count": float(leading_heading_fragment_count),
        "late_disclosure_heading_cluster": (
            leading_heading_fragment_count >= 3
            and (float(page_number) / float(safe_total_pages)) >= 0.55
            and not has_section_heading
            and not has_article_heading
            and not starts_with_table_of_contents
        ),
        "quoted_term_count": float(quoted_term_count),
        "quoted_term_ratio": float(quoted_term_count) / max(len(alpha_tokens), 1),
    }


def fit_tfidf_vectorizer(page_texts: list[str], *, max_features: int = 100) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(max_features=max_features)
    _ = vectorizer.fit(page_texts)
    return vectorizer


def _is_annex_context_page(feature_dict: FeatureDict) -> bool:
    return bool(
        feature_dict["has_annex_anchor"]
        or feature_dict["has_appendix_anchor"]
        or feature_dict["has_list_of_exhibits"]
        or feature_dict["has_list_of_schedules"]
        or feature_dict["has_definitions_heading"]
        or feature_dict["has_particulars_heading"]
        or float(feature_dict["annex_keyword_count"]) >= 3.0
    )


def _is_attached_subdocument_entry_page(feature_dict: FeatureDict) -> bool:
    relative_page = float(feature_dict["relative_page"])
    if relative_page < 0.45:
        return False
    return bool(
        feature_dict["has_annex_anchor"]
        or feature_dict["has_appendix_anchor"]
        or feature_dict["has_standalone_agreement_title"]
        or feature_dict["starts_with_table_of_contents"]
    )


def append_next_page_annex_feature(page_features: list[FeatureDict]) -> list[FeatureDict]:
    enriched_features: list[FeatureDict] = []
    last_annex_context_index: int | None = None
    annex_context_indices: list[int] = []
    last_attached_subdocument_index: int | None = None
    attached_subdocument_indices: list[int] = []
    for index, feature_dict in enumerate(page_features):
        enriched = dict(feature_dict)
        if _is_annex_context_page(feature_dict):
            last_annex_context_index = index
            annex_context_indices.append(index)
        if _is_attached_subdocument_entry_page(feature_dict):
            last_attached_subdocument_index = index
            attached_subdocument_indices.append(index)
        prev_page_has_annex_anchor = False
        prev_page_has_appendix_anchor = False
        prev_page_has_definitions_heading = False
        prev_page_has_section_heading = False
        prev_page_has_sig_block = False
        prev_page_has_attached_subdocument_entry = False
        prev_page_relative_page = 0.0
        if index > 0:
            prev_page_has_annex_anchor = bool(page_features[index - 1]["has_annex_anchor"])
            prev_page_has_appendix_anchor = bool(page_features[index - 1]["has_appendix_anchor"])
            prev_page_has_definitions_heading = bool(
                page_features[index - 1]["has_definitions_heading"]
            )
            prev_page_has_section_heading = bool(page_features[index - 1]["has_section_heading"])
            prev_page_has_sig_block = bool(page_features[index - 1]["has_sig_block"])
            prev_page_has_attached_subdocument_entry = _is_attached_subdocument_entry_page(
                page_features[index - 1]
            )
            prev_page_relative_page = float(page_features[index - 1]["relative_page"])
        next_page_has_annex_anchor = False
        next_page_has_appendix_anchor = False
        next_page_has_definitions_heading = False
        next_page_has_section_heading = False
        next_page_has_sig_block = False
        next_page_has_attached_subdocument_entry = False
        next_page_relative_page = 1.0
        if index + 1 < len(page_features):
            next_page_has_annex_anchor = bool(page_features[index + 1]["has_annex_anchor"])
            next_page_has_appendix_anchor = bool(page_features[index + 1]["has_appendix_anchor"])
            next_page_has_definitions_heading = bool(
                page_features[index + 1]["has_definitions_heading"]
            )
            next_page_has_section_heading = bool(page_features[index + 1]["has_section_heading"])
            next_page_has_sig_block = bool(page_features[index + 1]["has_sig_block"])
            next_page_has_attached_subdocument_entry = _is_attached_subdocument_entry_page(
                page_features[index + 1]
            )
            next_page_relative_page = float(page_features[index + 1]["relative_page"])
        annex_context_count_last_5 = sum(annex_index >= index - 4 for annex_index in annex_context_indices)
        has_prior_annex_context = last_annex_context_index is not None and last_annex_context_index < index
        attached_subdocument_count_last_8 = sum(
            attached_index >= index - 7 for attached_index in attached_subdocument_indices
        )
        has_prior_attached_subdocument = (
            last_attached_subdocument_index is not None and last_attached_subdocument_index < index
        )
        pages_since_last_attached_subdocument = (
            float(index - last_attached_subdocument_index)
            if has_prior_attached_subdocument and last_attached_subdocument_index is not None
            else float(len(page_features))
        )
        pages_since_last_annex_context = (
            float(index - last_annex_context_index)
            if has_prior_annex_context and last_annex_context_index is not None
            else float(len(page_features))
        )
        enriched["prev_page_has_annex_anchor"] = prev_page_has_annex_anchor
        enriched["prev_page_has_appendix_anchor"] = prev_page_has_appendix_anchor
        enriched["prev_page_has_definitions_heading"] = prev_page_has_definitions_heading
        enriched["prev_page_has_section_heading"] = prev_page_has_section_heading
        enriched["prev_page_has_sig_block"] = prev_page_has_sig_block
        enriched["prev_page_has_attached_subdocument_entry"] = prev_page_has_attached_subdocument_entry
        enriched["prev_page_relative_page"] = prev_page_relative_page
        enriched["has_prior_annex_context"] = has_prior_annex_context
        enriched["pages_since_last_annex_context"] = pages_since_last_annex_context
        enriched["recent_annex_context_count_5"] = float(annex_context_count_last_5)
        enriched["recent_annex_context_window_3"] = annex_context_count_last_5 >= 1 and pages_since_last_annex_context <= 3.0
        enriched["recent_annex_context_window_10"] = annex_context_count_last_5 >= 1 and pages_since_last_annex_context <= 10.0
        enriched["has_prior_attached_subdocument"] = has_prior_attached_subdocument
        enriched["pages_since_last_attached_subdocument"] = pages_since_last_attached_subdocument
        enriched["recent_attached_subdocument_count_8"] = float(attached_subdocument_count_last_8)
        enriched["recent_attached_subdocument_window_40"] = (
            attached_subdocument_count_last_8 >= 1 and pages_since_last_attached_subdocument <= 40.0
        )
        enriched["next_page_has_annex_anchor"] = next_page_has_annex_anchor
        enriched["next_page_has_appendix_anchor"] = next_page_has_appendix_anchor
        enriched["next_page_has_definitions_heading"] = next_page_has_definitions_heading
        enriched["next_page_has_section_heading"] = next_page_has_section_heading
        enriched["next_page_has_sig_block"] = next_page_has_sig_block
        enriched["next_page_has_attached_subdocument_entry"] = next_page_has_attached_subdocument_entry
        enriched["next_page_relative_page"] = next_page_relative_page
        enriched_features.append(enriched)
    return enriched_features


def build_agreement_documents(df: pd.DataFrame) -> list[AgreementDocument]:
    documents: list[AgreementDocument] = []
    for agreement_uuid, group in df.groupby("agreement_uuid", sort=False):
        documents.append(
            AgreementDocument(
                agreement_uuid=str(agreement_uuid),
                page_texts=[str(text) for text in group["page_text"].tolist()],
                labels=[str(label) for label in group["label"].tolist()],
            )
        )
    return documents


def build_feature_sequences(
    documents: list[AgreementDocument],
    vectorizer: TfidfVectorizer,
) -> list[list[FeatureDict]]:
    flattened_page_texts = [page_text for doc in documents for page_text in doc.page_texts]
    tfidf_matrix = cast(csr_matrix, vectorizer.transform(flattened_page_texts))
    feature_names = [
        f"tfidf_word_{feature_name}"
        for feature_name in vectorizer.get_feature_names_out().tolist()
    ]

    sequences: list[list[FeatureDict]] = []
    row_offset = 0
    for document in documents:
        total_pages = len(document.page_texts)
        page_features: list[FeatureDict] = []
        for page_number, page_text in enumerate(document.page_texts, start=1):
            features = extract_page_features(
                page_text=page_text,
                page_number=page_number,
                total_pages=total_pages,
            )
            tfidf_row = tfidf_matrix.getrow(row_offset)
            for column_index, score in zip(
                cast(np.ndarray, tfidf_row.indices),
                cast(np.ndarray, tfidf_row.data),
                strict=True,
            ):
                features[feature_names[int(column_index)]] = float(score)
            page_features.append(features)
            row_offset += 1
        sequences.append(append_next_page_annex_feature(page_features))
    return sequences


def build_label_sequences(documents: list[AgreementDocument]) -> list[list[str]]:
    return [document.labels for document in documents]


def flatten_label_sequences(label_sequences: list[list[str]]) -> list[str]:
    return [label for sequence in label_sequences for label in sequence]


def _build_agreement_level_metrics(
    documents: list[AgreementDocument],
    y_true: list[list[str]],
    y_pred: list[list[str]],
) -> dict[str, object]:
    if not (len(documents) == len(y_true) == len(y_pred)):
        raise ValueError("Agreement-level metrics inputs must have matching lengths.")

    exact_match_count = 0
    any_error_count = 0
    one_page_error_count = 0
    total_page_errors = 0
    error_distribution: dict[str, int] = {}
    agreement_summaries: list[dict[str, object]] = []

    for document, true_labels, predicted_labels in zip(documents, y_true, y_pred, strict=True):
        if len(true_labels) != len(predicted_labels):
            raise ValueError("Predicted and true label sequences must align by page.")
        page_error_count = sum(
            true_label != predicted_label
            for true_label, predicted_label in zip(true_labels, predicted_labels, strict=True)
        )
        total_page_errors += page_error_count
        error_distribution[str(page_error_count)] = error_distribution.get(str(page_error_count), 0) + 1
        if page_error_count == 0:
            exact_match_count += 1
        else:
            any_error_count += 1
        if page_error_count == 1:
            one_page_error_count += 1
        agreement_summaries.append(
            {
                "agreement_uuid": document.agreement_uuid,
                "page_count": len(true_labels),
                "page_error_count": page_error_count,
                "exact_match": page_error_count == 0,
            }
        )

    evaluated_agreement_count = len(documents)
    mean_page_errors = (
        float(total_page_errors) / float(evaluated_agreement_count)
        if evaluated_agreement_count > 0
        else 0.0
    )
    return {
        "evaluated_agreement_count": evaluated_agreement_count,
        "exact_match_agreement_count": exact_match_count,
        "exact_match_rate": (
            float(exact_match_count) / float(evaluated_agreement_count)
            if evaluated_agreement_count > 0
            else 0.0
        ),
        "agreements_with_any_page_error_count": any_error_count,
        "agreements_with_any_page_error_rate": (
            float(any_error_count) / float(evaluated_agreement_count)
            if evaluated_agreement_count > 0
            else 0.0
        ),
        "one_page_error_agreement_count": one_page_error_count,
        "one_page_error_agreement_rate": (
            float(one_page_error_count) / float(evaluated_agreement_count)
            if evaluated_agreement_count > 0
            else 0.0
        ),
        "mean_page_errors_per_agreement": mean_page_errors,
        "agreement_page_error_distribution": error_distribution,
        "agreement_summaries": sorted(
            agreement_summaries,
            key=lambda summary: (
                -cast(int, summary["page_error_count"]),
                cast(str, summary["agreement_uuid"]),
            ),
        ),
    }


def _split_targets(count: int, *, n_splits: int) -> dict[str, int]:
    if n_splits < 3:
        raise ValueError("n_splits must be at least 3 to build train/val/test splits.")
    train_fraction = float(n_splits - 2) / float(n_splits)
    holdout_fraction = 1.0 / float(n_splits)
    split_names = ["train", "val", "test"]
    raw_targets = {
        "train": count * train_fraction,
        "val": count * holdout_fraction,
        "test": count * holdout_fraction,
    }
    floor_targets = {split_name: int(math.floor(raw_targets[split_name])) for split_name in split_names}
    remainder = count - sum(floor_targets.values())
    if remainder > 0:
        split_order = sorted(
            split_names,
            key=lambda split_name: (raw_targets[split_name] - floor_targets[split_name], split_name),
            reverse=True,
        )
        for index in range(remainder):
            floor_targets[split_order[index % len(split_order)]] += 1
    return floor_targets


def _agreement_type_from_counts(*, sig_pages: int, back_pages: int) -> str:
    if sig_pages == 0:
        return "no_sig"
    if back_pages == 0:
        return "no_back_matter"
    if back_pages <= NORMAL_BACK_MATTER_MAX_PAGES:
        return "sig_plus_normal_back_matter"
    return "sig_plus_massive_back_matter"


def build_agreement_split_frame(df: pd.DataFrame) -> pd.DataFrame:
    announcement_dates = pd.to_datetime(df["date_announcement"], errors="raise")
    if announcement_dates.isna().any():
        raise ValueError("Found missing or invalid date_announcement values.")

    working = df[["agreement_uuid", "label"]].copy()
    working["agreement_uuid"] = working["agreement_uuid"].astype(str)
    working["announcement_year"] = announcement_dates.dt.year
    working["is_sig"] = (working["label"] == "sig").astype(int)
    working["is_back_matter"] = (working["label"] == "back_matter").astype(int)
    agreement_year_counts = cast(
        pd.Series,
        working.groupby("agreement_uuid")["announcement_year"].nunique(),
    )
    inconsistent = cast(pd.Series, agreement_year_counts[agreement_year_counts > 1])
    if not inconsistent.empty:
        raise ValueError(
            "Found agreements spanning multiple announcement years; cannot stratify by year."
        )

    agreement_meta = cast(
        pd.DataFrame,
        working.groupby("agreement_uuid", sort=True).agg(
            announcement_year=("announcement_year", "first"),
            sig_pages=("is_sig", "sum"),
            back_pages=("is_back_matter", "sum"),
            total_pages=("label", "size"),
        ),
    ).reset_index()
    agreement_meta["agreement_type"] = [
        _agreement_type_from_counts(sig_pages=int(sig_pages), back_pages=int(back_pages))
        for sig_pages, back_pages in zip(
            agreement_meta["sig_pages"],
            agreement_meta["back_pages"],
            strict=True,
        )
    ]
    return agreement_meta.sort_values("agreement_uuid").reset_index(drop=True)


def create_split_manifest(
    df: pd.DataFrame,
    *,
    split_path: Path,
    n_splits: int,
    test_fold_index: int,
    val_fold_index: int | None,
    split_seed: int,
) -> SplitManifest:
    agreement_meta = build_agreement_split_frame(df)
    agreement_count = len(agreement_meta)
    if agreement_count < n_splits:
        raise ValueError(
            f"Cannot create {n_splits} folds from only {agreement_count} agreements."
        )
    if test_fold_index < 0 or test_fold_index >= n_splits:
        raise ValueError(f"fold_index must be in [0, {n_splits - 1}].")
    resolved_val_fold_index = (
        (test_fold_index + 1) % n_splits
        if val_fold_index is None
        else val_fold_index
    )
    if resolved_val_fold_index < 0 or resolved_val_fold_index >= n_splits:
        raise ValueError(f"val_fold_index must be in [0, {n_splits - 1}].")
    if resolved_val_fold_index == test_fold_index:
        raise ValueError("val_fold_index must differ from fold_index.")

    rng = random.Random(split_seed)
    global_targets = _split_targets(agreement_count, n_splits=n_splits)

    year_counts_series = cast(pd.Series, agreement_meta["announcement_year"].value_counts())
    year_target_map = {
        int(year): _split_targets(int(count), n_splits=n_splits)
        for year, count in zip(
            year_counts_series.index.tolist(),
            year_counts_series.tolist(),
            strict=True,
        )
    }
    type_counts_series = cast(pd.Series, agreement_meta["agreement_type"].value_counts())
    type_target_map = {
        str(agreement_type): _split_targets(int(count), n_splits=n_splits)
        for agreement_type, count in zip(
            type_counts_series.index.tolist(),
            type_counts_series.tolist(),
            strict=True,
        )
    }

    split_names = ["train", "val", "test"]
    split_lists: dict[str, list[str]] = {split_name: [] for split_name in split_names}
    global_counts = {split_name: 0 for split_name in split_names}
    year_counts = {
        (int(year), split_name): 0
        for year in agreement_meta["announcement_year"].tolist()
        for split_name in split_names
    }
    type_counts = {
        (agreement_type, split_name): 0
        for agreement_type in agreement_meta["agreement_type"].tolist()
        for split_name in split_names
    }

    type_frequency = {
        str(agreement_type): int(count)
        for agreement_type, count in zip(
            type_counts_series.index.tolist(),
            type_counts_series.tolist(),
            strict=True,
        )
    }
    year_frequency = {
        int(year): int(count)
        for year, count in zip(
            year_counts_series.index.tolist(),
            year_counts_series.tolist(),
            strict=True,
        )
    }

    split_tiebreak = {
        "train": n_splits,
        "val": resolved_val_fold_index,
        "test": test_fold_index,
    }

    year_groups = [group.copy() for _, group in agreement_meta.groupby("announcement_year", sort=True)]
    year_groups.sort(
        key=lambda group: (len(group), int(cast(int, group["announcement_year"].iloc[0]))),
    )

    for year_group in year_groups:
        year_rows = [
            {
                "agreement_uuid": str(row["agreement_uuid"]),
                "announcement_year": int(row["announcement_year"]),
                "agreement_type": str(row["agreement_type"]),
                "back_pages": int(row["back_pages"]),
                "total_pages": int(row["total_pages"]),
            }
            for row in year_group.to_dict("records")
        ]
        rng.shuffle(year_rows)
        year_rows.sort(
            key=lambda row: (
                type_frequency[str(row["agreement_type"])],
                year_frequency[int(row["announcement_year"])],
                -int(row["back_pages"]),
                -int(row["total_pages"]),
                str(row["agreement_uuid"]),
            ),
        )

        for row in year_rows:
            agreement_uuid = str(row["agreement_uuid"])
            year = int(row["announcement_year"])
            agreement_type = str(row["agreement_type"])
            candidate_splits = [
                split_name
                for split_name in split_names
                if year_counts[(year, split_name)] < year_target_map[year][split_name]
            ]
            if not candidate_splits:
                candidate_splits = split_names

            def _score(split_name: str) -> tuple[int, int, int]:
                return (
                    type_counts[(agreement_type, split_name)] - type_target_map[agreement_type][split_name],
                    global_counts[split_name] - global_targets[split_name],
                    split_tiebreak[split_name],
                )

            chosen_split = min(candidate_splits, key=_score)
            split_lists[chosen_split].append(agreement_uuid)
            global_counts[chosen_split] += 1
            year_counts[(year, chosen_split)] += 1
            type_counts[(agreement_type, chosen_split)] += 1

    train_ids = sorted(split_lists["train"])
    val_ids = sorted(split_lists["val"])
    test_ids = sorted(split_lists["test"])
    split_map = {
        agreement_uuid: split_name
        for split_name, split_ids in (
            ("train", train_ids),
            ("val", val_ids),
            ("test", test_ids),
        )
        for agreement_uuid in split_ids
    }
    agreement_details = [
        {
            "agreement_uuid": str(row["agreement_uuid"]),
            "split": split_map[str(row["agreement_uuid"])],
            "announcement_year": int(row["announcement_year"]),
            "agreement_type": str(row["agreement_type"]),
            "sig_pages": int(row["sig_pages"]),
            "back_pages": int(row["back_pages"]),
            "total_pages": int(row["total_pages"]),
        }
        for row in agreement_meta.to_dict("records")
    ]
    agreement_type_counts_by_split = {
        split_name: {
            agreement_type: int(
                (
                    (agreement_meta["agreement_type"] == agreement_type)
                    & (agreement_meta["agreement_uuid"].isin(split_lists[split_name]))
                ).sum()
            )
            for agreement_type in AGREEMENT_TYPE_LABELS
        }
        for split_name in ("train", "val", "test")
    }
    split_series = pd.Series(
        [split_map[str(agreement_uuid)] for agreement_uuid in agreement_meta["agreement_uuid"].tolist()],
        index=agreement_meta.index,
    )
    year_counts_table = pd.crosstab(
        agreement_meta["announcement_year"],
        split_series,
    )
    year_counts_by_split: dict[str, dict[str, int]] = {}
    for row_index, year_value in enumerate(year_counts_table.index.tolist()):
        year_key = str(int(year_value))
        year_row = cast(pd.Series, year_counts_table.iloc[row_index])
        year_counts_by_split[year_key] = {}
        for split_name in ("train", "val", "test"):
            if split_name in year_row.index:
                year_counts_by_split[year_key][split_name] = int(cast(int, year_row[split_name]))
            else:
                year_counts_by_split[year_key][split_name] = 0

    manifest: SplitManifest = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "meta": {
            "n_splits": n_splits,
            "test_fold_index": test_fold_index,
            "val_fold_index": resolved_val_fold_index,
            "split_seed": split_seed,
            "stratification_strategy": "announcement_year_and_agreement_type",
        },
        "details": {
            "split_counts": {
                "train": len(train_ids),
                "val": len(val_ids),
                "test": len(test_ids),
            },
            "agreement_type_counts_by_split": agreement_type_counts_by_split,
            "year_counts_by_split": year_counts_by_split,
            "agreement_details": agreement_details,
        },
    }
    split_path.parent.mkdir(parents=True, exist_ok=True)
    _ = split_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def load_split_manifest(split_path: Path) -> SplitManifest:
    manifest_raw = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_raw, dict):
        raise ValueError("Split manifest must be a mapping.")
    required_keys = {"train", "val", "test", "meta", "details"}
    missing = required_keys - set(manifest_raw)
    if missing:
        raise ValueError(f"Split manifest missing keys: {sorted(missing)}")

    train_ids = [str(agreement_id) for agreement_id in cast(list[object], manifest_raw["train"])]
    val_ids = [str(agreement_id) for agreement_id in cast(list[object], manifest_raw["val"])]
    test_ids = [str(agreement_id) for agreement_id in cast(list[object], manifest_raw["test"])]
    meta_raw = cast(dict[str, object], manifest_raw["meta"])
    details_raw = _coerce_dict(manifest_raw["details"], field_name="details")
    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "meta": {
            "n_splits": _coerce_int(meta_raw["n_splits"], field_name="n_splits"),
            "test_fold_index": _coerce_int(
                meta_raw["test_fold_index"],
                field_name="test_fold_index",
            ),
            "val_fold_index": _coerce_int(
                meta_raw["val_fold_index"],
                field_name="val_fold_index",
            ),
            "split_seed": _coerce_int(meta_raw["split_seed"], field_name="split_seed"),
            "stratification_strategy": _coerce_str(
                meta_raw["stratification_strategy"],
                field_name="stratification_strategy",
            ),
        },
        "details": details_raw,
    }


def load_or_create_split_manifest(
    df: pd.DataFrame,
    *,
    split_path: Path,
    n_splits: int,
    test_fold_index: int,
    val_fold_index: int | None,
    split_seed: int,
) -> SplitManifest:
    if split_path.exists():
        manifest = load_split_manifest(split_path)
        manifest_meta = manifest["meta"]
        expected_val_fold_index = (
            (test_fold_index + 1) % manifest_meta["n_splits"]
            if val_fold_index is None
            else val_fold_index
        )
        if (
            manifest_meta["n_splits"] != n_splits
            or manifest_meta["test_fold_index"] != test_fold_index
            or manifest_meta["val_fold_index"] != expected_val_fold_index
            or manifest_meta["split_seed"] != split_seed
            or manifest_meta["stratification_strategy"] != "announcement_year_and_agreement_type"
        ):
            raise ValueError(
                f"Existing split manifest at {split_path} does not match the requested fold "
                + "configuration. Delete the file to regenerate it."
            )
    else:
        manifest = create_split_manifest(
            df,
            split_path=split_path,
            n_splits=n_splits,
            test_fold_index=test_fold_index,
            val_fold_index=val_fold_index,
            split_seed=split_seed,
        )

    current_agreement_ids = set(
        str(agreement_id) for agreement_id in df["agreement_uuid"].drop_duplicates().tolist()
    )
    split_id_sets = {
        split_name: set(manifest[split_name])
        for split_name in ("train", "val", "test")
    }
    if split_id_sets["train"] & split_id_sets["val"]:
        raise ValueError("Split manifest has overlap between train and val agreements.")
    if split_id_sets["train"] & split_id_sets["test"]:
        raise ValueError("Split manifest has overlap between train and test agreements.")
    if split_id_sets["val"] & split_id_sets["test"]:
        raise ValueError("Split manifest has overlap between val and test agreements.")

    manifest_ids = split_id_sets["train"] | split_id_sets["val"] | split_id_sets["test"]
    if manifest_ids != current_agreement_ids:
        raise ValueError(
            f"Split manifest at {split_path} does not cover the current dataset exactly."
        )
    return manifest


def split_dataframe_from_manifest(
    df: pd.DataFrame,
    manifest: SplitManifest,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = cast(pd.DataFrame, df[df["agreement_uuid"].isin(manifest["train"])].copy())
    val_df = cast(pd.DataFrame, df[df["agreement_uuid"].isin(manifest["val"])].copy())
    test_df = cast(pd.DataFrame, df[df["agreement_uuid"].isin(manifest["test"])].copy())
    return train_df, val_df, test_df


def enforce_monotonic_transition_weights(
    crf_model: CRFModelProtocol,
    *,
    illegal_transition_weight: float,
) -> dict[tuple[str, str], float]:
    transition_features = crf_model.transition_features_

    overwritten: dict[tuple[str, str], float] = {}
    for source_label in PAGE_LABELS:
        for target_label in PAGE_LABELS:
            if LABEL_TO_INDEX[target_label] >= LABEL_TO_INDEX[source_label]:
                continue
            transition = (source_label, target_label)
            transition_features[transition] = illegal_transition_weight
            overwritten[transition] = illegal_transition_weight
    return overwritten


def _save_report(report_path: Path, report_text: str) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _ = report_path.write_text(report_text, encoding="utf-8")


def _save_metrics(metrics_path: Path, metrics_payload: dict[str, object]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    _ = metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _run_output_tag(*, mode: str, fit_split: str, eval_split: str) -> str:
    if mode == "tune":
        return "tune_val"
    if fit_split == "train_val" and eval_split == "test":
        return "final_test"
    if fit_split == "train" and eval_split == "val":
        return "dev_val"
    return f"{mode}_{fit_split}_{eval_split}"


def _resolve_run_output_path(
    path: Path,
    *,
    default_path: Path,
    mode: str,
    fit_split: str,
    eval_split: str,
) -> Path:
    if path != default_path:
        return path
    tag = _run_output_tag(mode=mode, fit_split=fit_split, eval_split=eval_split)
    return default_path.with_name(f"{default_path.stem}_{tag}{default_path.suffix}")


def _load_best_optuna_hyperparameters(optuna_path: Path) -> CRFHyperparameters:
    if not optuna_path.exists():
        raise ValueError(
            f"Optuna summary file not found at {optuna_path}. Run tune mode first or omit "
            + "--use-best-optuna-params."
        )
    payload = json.loads(optuna_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Optuna summary must be a mapping.")
    best_params = payload.get("best_params")
    if not isinstance(best_params, dict):
        raise ValueError("Optuna summary missing `best_params`.")
    if {"c1", "c2", "tfidf_max_features"} - set(best_params):
        raise ValueError("Optuna `best_params` must contain c1, c2, and tfidf_max_features.")

    c1_value = best_params["c1"]
    c2_value = best_params["c2"]
    tfidf_value = best_params["tfidf_max_features"]
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


def _run_crf_experiment(
    *,
    train_documents: list[AgreementDocument],
    eval_documents: list[AgreementDocument],
    split_manifest: SplitManifest,
    fit_split: str,
    eval_split: str,
    illegal_transition_weight: float,
    tfidf_max_features: int,
    c1: float,
    c2: float,
    max_iterations: int = 100,
    log_prefix: str | None = None,
) -> dict[str, object]:
    prefix = f"{log_prefix} " if log_prefix else ""
    train_page_texts = [page_text for doc in train_documents for page_text in doc.page_texts]
    _log(
        f"{prefix}[crf] preparing features: train_agreements={len(train_documents)}, "
        + f"eval_agreements={len(eval_documents)}, tfidf_max_features={tfidf_max_features}"
    )
    vectorizer = fit_tfidf_vectorizer(train_page_texts, max_features=tfidf_max_features)
    x_train = build_feature_sequences(train_documents, vectorizer)
    y_train = build_label_sequences(train_documents)
    x_eval = build_feature_sequences(eval_documents, vectorizer)
    y_eval = build_label_sequences(eval_documents)

    sklearn_crfsuite, crf_metrics = _load_crf_modules()
    crf_model = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=c1,
        c2=c2,
        max_iterations=max_iterations,
        all_possible_transitions=True,
    )
    _log(
        f"{prefix}[crf] fitting model: c1={c1:.6f}, c2={c2:.6f}, max_iterations={max_iterations}"
    )
    _ = crf_model.fit(x_train, y_train)
    _log(f"{prefix}[crf] fit complete")

    overwritten_transitions = enforce_monotonic_transition_weights(
        crf_model,
        illegal_transition_weight=illegal_transition_weight,
    )
    _log(
        f"{prefix}[crf] predicting {sum(len(doc.labels) for doc in eval_documents)} pages "
        + f"across {len(eval_documents)} agreements"
    )
    y_pred = crf_model.predict(x_eval)
    y_eval_flat = flatten_label_sequences(y_eval)
    y_pred_flat = flatten_label_sequences(y_pred)
    report_text = crf_metrics.flat_classification_report(
        y_eval,
        y_pred,
        labels=PAGE_LABELS,
        digits=4,
    )
    precision, recall, f1, support = cast(
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        precision_recall_fscore_support(
            y_eval_flat,
            y_pred_flat,
            labels=PAGE_LABELS,
            zero_division=cast(str, cast(object, 0)),
        ),
    )
    confusion = confusion_matrix(
        y_eval_flat,
        y_pred_flat,
        labels=PAGE_LABELS,
    )
    page_accuracy = (
        float(sum(true_label == predicted_label for true_label, predicted_label in zip(y_eval_flat, y_pred_flat, strict=True)))
        / float(len(y_eval_flat))
        if y_eval_flat
        else 0.0
    )
    agreement_level_metrics = _build_agreement_level_metrics(eval_documents, y_eval, y_pred)
    report_text = (
        report_text
        + "\nAgreement-level exact match\n"
        + f"exact_match_agreements: {agreement_level_metrics['exact_match_agreement_count']}"
        + f" / {agreement_level_metrics['evaluated_agreement_count']}\n"
        + "agreements_with_any_page_error: "
        + f"{agreement_level_metrics['agreements_with_any_page_error_count']}"
        + f" / {agreement_level_metrics['evaluated_agreement_count']}\n"
        + "agreements_with_exactly_one_page_error: "
        + f"{agreement_level_metrics['one_page_error_agreement_count']}"
        + f" / {agreement_level_metrics['evaluated_agreement_count']}\n"
    )
    metrics_payload: dict[str, object] = {
        "labels": PAGE_LABELS,
        "accuracy": page_accuracy,
        "macro_f1": float(np.mean(f1)),
        "flat_classification_report": report_text,
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(PAGE_LABELS)
        },
        "confusion_matrix": {
            "labels": PAGE_LABELS,
            "matrix": confusion.tolist(),
        },
        "split_meta": split_manifest["meta"],
        "split_counts": split_manifest["details"]["split_counts"],
        "agreement_level": agreement_level_metrics,
        "run_configuration": {
            "fit_split": fit_split,
            "eval_split": eval_split,
            "tfidf_max_features": tfidf_max_features,
            "c1": c1,
            "c2": c2,
            "max_iterations": max_iterations,
        },
    }
    _log(
        f"{prefix}[crf] evaluation complete: accuracy={page_accuracy:.4f}, "
        + f"macro_f1={float(np.mean(f1)):.4f}"
    )
    return {
        "crf_model": crf_model,
        "vectorizer": vectorizer,
        "overwritten_transitions": overwritten_transitions,
        "report_text": report_text,
        "metrics_payload": metrics_payload,
    }


def _save_artifact(
    model_path: Path,
    *,
    crf_model: CRFModelProtocol,
    vectorizer: TfidfVectorizer,
    train_agreement_ids: list[str],
    val_agreement_ids: list[str],
    test_agreement_ids: list[str],
    fitted_agreement_ids: list[str],
    overwritten_transitions: dict[tuple[str, str], float],
    label_violations: list[LabelViolation],
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "crf_model": crf_model,
        "vectorizer": vectorizer,
        "labels": PAGE_LABELS,
        "train_agreement_ids": train_agreement_ids,
        "val_agreement_ids": val_agreement_ids,
        "test_agreement_ids": test_agreement_ids,
        "fitted_agreement_ids": fitted_agreement_ids,
        "overwritten_transitions": overwritten_transitions,
        "label_violations": [
            {
                "agreement_uuid": violation.agreement_uuid,
                "previous_label": violation.previous_label,
                "current_label": violation.current_label,
                "previous_order": violation.previous_order,
                "current_order": violation.current_order,
            }
            for violation in label_violations
        ],
    }
    _ = joblib.dump(artifact, model_path)


def train_and_evaluate(
    *,
    data_path: Path,
    split_path: Path,
    model_path: Path,
    report_path: Path,
    metrics_path: Path,
    optuna_path: Path,
    n_splits: int,
    fold_index: int,
    val_fold_index: int | None,
    split_seed: int,
    fit_split: str,
    eval_split: str,
    use_best_optuna_params: bool,
    illegal_transition_weight: float,
) -> dict[str, object]:
    df = load_page_dataframe(data_path)
    _log(f"[crf] loaded dataset: rows={len(df)}, agreements={df['agreement_uuid'].nunique()}")
    label_violations = find_label_monotonicity_violations(df)
    if label_violations:
        _log(
            "Detected backward gold-label transitions in "
            + f"{len(label_violations)} agreements. "
            + "The CRF constraint will still enforce monotonic predictions, "
            + "but the labels should be reviewed."
        )

    split_manifest = load_or_create_split_manifest(
        df,
        split_path=split_path,
        n_splits=n_splits,
        test_fold_index=fold_index,
        val_fold_index=val_fold_index,
        split_seed=split_seed,
    )
    train_df, val_df, test_df = split_dataframe_from_manifest(df, split_manifest)
    _log(
        "[crf] split counts: "
        + f"train={train_df['agreement_uuid'].nunique()}, "
        + f"val={val_df['agreement_uuid'].nunique()}, "
        + f"test={test_df['agreement_uuid'].nunique()}"
    )
    train_agreement_ids = split_manifest["train"]
    val_agreement_ids = split_manifest["val"]
    test_agreement_ids = split_manifest["test"]
    if fit_split == "train":
        fit_df = train_df.copy()
        fitted_agreement_ids = list(train_agreement_ids)
    elif fit_split == "train_val":
        fit_df = pd.concat([train_df, val_df], ignore_index=True)
        fitted_agreement_ids = sorted(train_agreement_ids + val_agreement_ids)
    else:
        raise ValueError(f"Unsupported fit_split: {fit_split}")

    if eval_split == "val":
        eval_df = val_df.copy()
        eval_agreement_ids = val_agreement_ids
    elif eval_split == "test":
        eval_df = test_df.copy()
        eval_agreement_ids = test_agreement_ids
    else:
        raise ValueError(f"Unsupported eval_split: {eval_split}")

    if fit_split == "train_val" and eval_split == "val":
        raise ValueError("Cannot evaluate on `val` after fitting on `train_val`.")

    if use_best_optuna_params:
        selected_hyperparameters = _load_best_optuna_hyperparameters(optuna_path)
    else:
        selected_hyperparameters: CRFHyperparameters = {
            "tfidf_max_features": 100,
            "c1": 0.1,
            "c2": 0.1,
        }
    _log(
        f"[crf] selected hyperparameters: c1={selected_hyperparameters['c1']:.6f}, "
        + f"c2={selected_hyperparameters['c2']:.6f}, "
        + f"tfidf_max_features={selected_hyperparameters['tfidf_max_features']}, "
        + f"fit_split={fit_split}, eval_split={eval_split}"
    )

    train_documents = build_agreement_documents(fit_df)
    eval_documents = build_agreement_documents(eval_df)
    experiment = _run_crf_experiment(
        train_documents=train_documents,
        eval_documents=eval_documents,
        split_manifest=split_manifest,
        fit_split=fit_split,
        eval_split=eval_split,
        illegal_transition_weight=illegal_transition_weight,
        tfidf_max_features=selected_hyperparameters["tfidf_max_features"],
        c1=selected_hyperparameters["c1"],
        c2=selected_hyperparameters["c2"],
        log_prefix="[crf][train]",
    )
    report_text = cast(str, experiment["report_text"])
    metrics_payload = cast(dict[str, object], experiment["metrics_payload"])
    metrics_payload["fit_counts"] = {
        "train_agreement_count": len(train_agreement_ids),
        "val_agreement_count": len(val_agreement_ids),
        "test_agreement_count": len(test_agreement_ids),
        "fitted_agreement_count": len(fitted_agreement_ids),
        "evaluated_agreement_count": len(eval_agreement_ids),
    }
    if use_best_optuna_params:
        metrics_payload["optuna"] = {
            "best_params": selected_hyperparameters,
            "source": str(optuna_path),
        }

    _log(f"[crf] writing report to {report_path}")
    _save_report(report_path, report_text)
    _log(f"[crf] writing metrics to {metrics_path}")
    _save_metrics(metrics_path, metrics_payload)
    _save_artifact(
        model_path,
        crf_model=cast(CRFModelProtocol, experiment["crf_model"]),
        vectorizer=cast(TfidfVectorizer, experiment["vectorizer"]),
        train_agreement_ids=train_agreement_ids,
        val_agreement_ids=val_agreement_ids,
        test_agreement_ids=test_agreement_ids,
        fitted_agreement_ids=fitted_agreement_ids,
        overwritten_transitions=cast(
            dict[tuple[str, str], float],
            experiment["overwritten_transitions"],
        ),
        label_violations=label_violations,
    )
    _log(f"[crf] saved artifact to {model_path}")
    print(report_text)

    return {
        "report_text": report_text,
        "metrics_path": str(metrics_path),
        "train_agreement_count": len(train_agreement_ids),
        "val_agreement_count": len(val_agreement_ids),
        "test_agreement_count": len(test_agreement_ids),
        "fitted_agreement_count": len(fitted_agreement_ids),
        "evaluated_agreement_count": len(eval_agreement_ids),
        "label_violation_count": len(label_violations),
        "used_best_optuna_params": use_best_optuna_params,
        "overwritten_transition_count": len(
            cast(dict[tuple[str, str], float], experiment["overwritten_transitions"])
        ),
    }


def tune_hyperparameters(
    *,
    data_path: Path,
    split_path: Path,
    model_path: Path,
    report_path: Path,
    metrics_path: Path,
    optuna_path: Path,
    n_splits: int,
    fold_index: int,
    val_fold_index: int | None,
    split_seed: int,
    illegal_transition_weight: float,
    optuna_trials: int,
) -> dict[str, object]:
    if optuna_trials <= 0:
        raise ValueError("optuna_trials must be positive.")

    df = load_page_dataframe(data_path)
    _log(f"[crf] loaded dataset: rows={len(df)}, agreements={df['agreement_uuid'].nunique()}")
    label_violations = find_label_monotonicity_violations(df)
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
        "[crf] tuning split counts: "
        + f"train={train_df['agreement_uuid'].nunique()}, "
        + f"val={val_df['agreement_uuid'].nunique()}, "
        + f"trials={optuna_trials}"
    )
    train_agreement_ids = split_manifest["train"]
    val_agreement_ids = split_manifest["val"]
    test_agreement_ids = split_manifest["test"]
    train_documents = build_agreement_documents(train_df)
    val_documents = build_agreement_documents(val_df)

    def _objective(trial: optuna.Trial) -> float:
        c1 = float(trial.suggest_float("c1", 1e-3, 1.0, log=True))
        c2 = float(trial.suggest_float("c2", 1e-3, 1.0, log=True))
        tfidf_max_features = int(trial.suggest_int("tfidf_max_features", 50, 250, step=25))
        _log(
            f"[crf][trial {trial.number + 1}/{optuna_trials}] "
            + f"c1={c1:.6f} c2={c2:.6f} tfidf_max_features={tfidf_max_features}"
        )
        experiment = _run_crf_experiment(
            train_documents=train_documents,
            eval_documents=val_documents,
            split_manifest=split_manifest,
            fit_split="train",
            eval_split="val",
            illegal_transition_weight=illegal_transition_weight,
            tfidf_max_features=tfidf_max_features,
            c1=c1,
            c2=c2,
            log_prefix=f"[crf][trial {trial.number + 1}/{optuna_trials}]",
        )
        metrics_payload = cast(dict[str, object], experiment["metrics_payload"])
        macro_f1 = float(cast(float, metrics_payload["macro_f1"]))
        trial.set_user_attr("macro_f1", macro_f1)
        trial.set_user_attr("c1", c1)
        trial.set_user_attr("c2", c2)
        trial.set_user_attr("tfidf_max_features", tfidf_max_features)
        _log(f"[crf][trial {trial.number + 1}/{optuna_trials}] macro_f1={macro_f1:.4f}")
        return macro_f1

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=split_seed),
    )
    _log("[crf] starting Optuna search")
    study.optimize(_objective, n_trials=optuna_trials)
    best_c1 = float(cast(float, study.best_params["c1"]))
    best_c2 = float(cast(float, study.best_params["c2"]))
    best_tfidf_max_features = int(cast(int, study.best_params["tfidf_max_features"]))
    best_params: dict[str, object] = {
        "c1": best_c1,
        "c2": best_c2,
        "tfidf_max_features": best_tfidf_max_features,
    }

    best_experiment = _run_crf_experiment(
        train_documents=train_documents,
        eval_documents=val_documents,
        split_manifest=split_manifest,
        fit_split="train",
        eval_split="val",
        illegal_transition_weight=illegal_transition_weight,
        tfidf_max_features=best_tfidf_max_features,
        c1=best_c1,
        c2=best_c2,
        log_prefix="[crf][best]",
    )
    report_text = cast(str, best_experiment["report_text"])
    metrics_payload = cast(dict[str, object], best_experiment["metrics_payload"])
    metrics_payload["fit_counts"] = {
        "train_agreement_count": len(train_agreement_ids),
        "val_agreement_count": len(val_agreement_ids),
        "test_agreement_count": len(test_agreement_ids),
        "fitted_agreement_count": len(train_agreement_ids),
        "evaluated_agreement_count": len(val_agreement_ids),
    }
    metrics_payload["optuna"] = {
        "best_value": float(study.best_value),
        "best_params": best_params,
        "n_trials": optuna_trials,
    }
    best_summary = {
        "best_value": float(study.best_value),
        "best_params": best_params,
        "n_trials": optuna_trials,
        "split_meta": split_manifest["meta"],
        "run_configuration": {
            "fit_split": "train",
            "eval_split": "val",
        },
    }

    _log(
        f"[crf] best trial complete: macro_f1={float(study.best_value):.4f}, "
        + f"c1={best_c1:.6f}, c2={best_c2:.6f}, tfidf_max_features={best_tfidf_max_features}"
    )
    _log(f"[crf] writing report to {report_path}")
    _save_report(report_path, report_text)
    _log(f"[crf] writing metrics to {metrics_path}")
    _save_metrics(metrics_path, metrics_payload)
    optuna_path.parent.mkdir(parents=True, exist_ok=True)
    _ = optuna_path.write_text(
        json.dumps(best_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _log(f"[crf] wrote Optuna summary to {optuna_path}")
    _save_artifact(
        model_path,
        crf_model=cast(CRFModelProtocol, best_experiment["crf_model"]),
        vectorizer=cast(TfidfVectorizer, best_experiment["vectorizer"]),
        train_agreement_ids=train_agreement_ids,
        val_agreement_ids=val_agreement_ids,
        test_agreement_ids=test_agreement_ids,
        fitted_agreement_ids=list(train_agreement_ids),
        overwritten_transitions=cast(
            dict[tuple[str, str], float],
            best_experiment["overwritten_transitions"],
        ),
        label_violations=label_violations,
    )
    _log(f"[crf] saved artifact to {model_path}")
    print(report_text)

    return {
        "report_text": report_text,
        "metrics_path": str(metrics_path),
        "optuna_path": str(optuna_path),
        "best_value": float(study.best_value),
        "best_params": best_params,
    }


def main() -> None:
    args = parse_args()
    resolved_model_path = _resolve_run_output_path(
        args.model_path,
        default_path=DEFAULT_MODEL_PATH,
        mode=args.mode,
        fit_split=args.fit_split if args.mode == "train" else "train",
        eval_split=args.eval_split if args.mode == "train" else "val",
    )
    resolved_report_path = _resolve_run_output_path(
        args.report_path,
        default_path=DEFAULT_REPORT_PATH,
        mode=args.mode,
        fit_split=args.fit_split if args.mode == "train" else "train",
        eval_split=args.eval_split if args.mode == "train" else "val",
    )
    resolved_metrics_path = _resolve_run_output_path(
        args.metrics_path,
        default_path=DEFAULT_METRICS_PATH,
        mode=args.mode,
        fit_split=args.fit_split if args.mode == "train" else "train",
        eval_split=args.eval_split if args.mode == "train" else "val",
    )
    if args.mode == "train":
        _ = train_and_evaluate(
            data_path=args.data_path,
            split_path=args.split_path,
            model_path=resolved_model_path,
            report_path=resolved_report_path,
            metrics_path=resolved_metrics_path,
            optuna_path=args.optuna_path,
            n_splits=args.n_splits,
            fold_index=args.fold_index,
            val_fold_index=args.val_fold_index,
            split_seed=args.split_seed,
            fit_split=args.fit_split,
            eval_split=args.eval_split,
            use_best_optuna_params=args.use_best_optuna_params,
            illegal_transition_weight=args.illegal_transition_weight,
        )
    else:
        _ = tune_hyperparameters(
            data_path=args.data_path,
            split_path=args.split_path,
            model_path=resolved_model_path,
            report_path=resolved_report_path,
            metrics_path=resolved_metrics_path,
            optuna_path=args.optuna_path,
            n_splits=args.n_splits,
            fold_index=args.fold_index,
            val_fold_index=args.val_fold_index,
            split_seed=args.split_seed,
            illegal_transition_weight=args.illegal_transition_weight,
            optuna_trials=args.optuna_trials,
        )


if __name__ == "__main__":
    main()
