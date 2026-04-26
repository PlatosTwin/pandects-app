"""
Feature extraction utilities for page classification.

This module provides functions to extract features from text and HTML content
for use in page classification models.
"""
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false

import re
import string
from collections.abc import Sequence
from typing import cast

from joblib import Parallel, delayed
import numpy as np
import pandas as pd


BOILERPLATE_TERMS = ["hereto", "herein", "hereby", "thereof", "wherein"]
LEGAL_KEYWORDS = [
    "table of contents",
    "execution version",
    "in witness whereof",
    "exhibit",
    "signature",
    "list of exhibits",
    "schedule",
    "list of schedules",
    "index of",
    "recitals",
    "whereas",
    "now, therefore",
    "signed",
    "execution date",
    "effective",
    "dated as of",
    "entered into by and among",
    "[signature",
    "w i t n e s e t h",
    "w i t n e s s e t h",
    "/s/",
    "intentionally blank",
    "page follows]",
    "page follows.]",
    "by:",
]

FEATURE_NAMES = [
    "num_words",
    "num_chars",
    "avg_chars_per_word",
    "prop_spaces",
    "prop_digits",
    "prop_newlines",
    "prop_punct",
    "num_lines",
    "avg_line_len",
    "frac_short_lines",
    "frac_empty_lines",
    "sig_line_count",
    "frac_heading_lines",
    "count_article_10",
    "count_section_10",
    "flag_is_all_digits",
    "flag_is_less_than_order",
    "count_section",
    "count_article",
    "num_all_caps",
    "prop_word_cap",
    "num_bigrams",
    "unique_bigrams",
    "prop_unique_bigrams",
    "num_trigrams",
    "unique_trigrams",
    "prop_unique_trigrams",
    "num_tags",
    "tag_to_text_ratio",
    "link_count",
    "img_count",
    "heading_tags",
    "list_count",
    "bullet_count",
    "count_hereto",
    "count_herein",
    "count_hereby",
    "count_thereof",
    "count_wherein",
    "kw_table_of_contents",
    "kw_execution_version",
    "kw_in_witness_whereof",
    "kw_exhibit",
    "kw_signature",
    "kw_list_of_exhibits",
    "kw_schedule",
    "kw_list_of_schedules",
    "kw_index_of",
    "kw_recitals",
    "kw_whereas",
    "kw_now_therefore",
    "kw_signed",
    "kw_execution_date",
    "kw_effective",
    "kw_dated_as_of",
    "kw_entered_into_by_and_among",
    "kw_bracket_signature",
    "kw_w_i_t_n_e_s_e_t_h",
    "kw_slash_s",
    "kw_intentionally_blank",
    "kw_page_follows",
    "kw_page_follows_dot",
    "kw_by_colon",
    "sig_indicators",
    "prop_colon",
    "prop_period",
    "num_consecutive_periods",
    "prop_comma",
    "has_table",
    "count_p",
    "count_div",
    "order",
    "exhibit_heading",
    "has_counterparts",
    "has_form_of",
    "signature_page_to",
    "sig_field_cues",
    "underscore_runs",
    "page_index",
    "doc_page_count",
    "rel_page_pos",
    "pages_remaining",
    "is_last_page",
    "in_last_three_pages",
    "in_last_decile",
    "in_last_quintile",
    "has_executed_as_deed",
    "has_signature_of_witness",
    "has_witness_address",
    "has_witness_occupation",
    "has_confidential_treatment",
    "has_redaction_marker",
    "witness_block_combo",
]

NEIGHBOR_CONTEXT_SOURCE_FEATURES = [
    "kw_exhibit",
    "kw_schedule",
    "kw_slash_s",
    "sig_field_cues",
    "has_executed_as_deed",
    "has_signature_of_witness",
    "num_words",
    "prop_digits",
    "rel_page_pos",
    "in_last_quintile",
]

_SIG_LINE_RE = re.compile(r"^(by|name|title)\s*:", re.IGNORECASE)
_ARTICLE_RE = re.compile(r"\barticle\b", re.IGNORECASE)
_SECTION_RE = re.compile(r"\bsection\b", re.IGNORECASE)
_DIGIT_RE = re.compile(r"^[\-\s-]*(\d+)[\-\s-]*$")
_CONSEC_PERIOD_RE = re.compile(r"(?: *\.)+")
_HAS_TABLE_RE = re.compile(r"</?(table|tr|td)")
_EXHIBIT_HEADING_RE = re.compile(
    r"^\s*(exhibit|annex|schedule)\s+([a-z0-9][a-z0-9\.\-\(\)]*)",
    re.IGNORECASE,
)
_UNDERSCORE_RUN_RE = re.compile(r"_{5,}")
_REDACTION_RE = re.compile(r"\[\*{3,}\]")


def feature_name_list() -> list[str]:
    return list(FEATURE_NAMES)


def neighbor_context_feature_name_list() -> list[str]:
    names: list[str] = []
    for feature in NEIGHBOR_CONTEXT_SOURCE_FEATURES:
        names.append(f"prev_{feature}")
        names.append(f"next_{feature}")
    names.extend(
        [
            "delta_prev_num_words",
            "delta_next_num_words",
            "delta_prev_rel_page_pos",
            "delta_next_rel_page_pos",
        ]
    )
    return names


def full_feature_name_list(include_neighbor_context: bool) -> list[str]:
    names = feature_name_list()
    if include_neighbor_context:
        names.extend(neighbor_context_feature_name_list())
    return names


def extract_features(
    text: str,
    html: str,
    order: float,
    page_index: float,
    doc_page_count: float,
) -> np.ndarray:
    """
    Extract features from text and HTML content for page classification.
    
    Args:
        text: Raw text content of the page
        html: HTML content of the page
        order: Page order/sequence number
        page_index: Zero-based page index within the agreement
        doc_page_count: Total pages in the agreement
        
    Returns:
        numpy.ndarray: Feature vector for classification
    """
    # Handle NaN values
    text = str(text) if text == text else ""
    html = str(html) if html == html else ""
    safe_doc_page_count = max(float(doc_page_count), 1.0)
    safe_page_index = max(float(page_index), 0.0)

    # Basic text statistics
    num_chars = len(text)
    words = text.split()
    num_words = len(words)
    avg_chars_per_word = num_chars / num_words if num_words > 0 else 0.0
    prop_spaces = text.count(" ") / num_chars if num_chars > 0 else 0.0
    prop_digits = sum(c.isdigit() for c in text) / num_chars if num_chars > 0 else 0.0
    prop_newlines = text.count("\n") / num_chars if num_chars > 0 else 0.0
    prop_punct = (
        sum(c in string.punctuation for c in text) / num_chars if num_chars > 0 else 0.0
    )

    # Line-structure stats
    lines = text.split("\n")
    num_lines = len(lines)
    line_lengths = [len(line.strip()) for line in lines]
    avg_line_len = sum(line_lengths) / num_lines if num_lines > 0 else 0.0
    very_short_lines = sum(1 for length in line_lengths if length <= 3)
    empty_lines = sum(1 for length in line_lengths if length == 0)
    frac_short_lines = very_short_lines / num_lines if num_lines > 0 else 0.0
    frac_empty_lines = empty_lines / num_lines if num_lines > 0 else 0.0

    # Signature block line cues
    sig_line_count = sum(1 for line in lines if _SIG_LINE_RE.match(line.strip()))

    # Title/heading density
    non_empty_lines = [line for line in lines if line.strip()]
    non_empty_count = len(non_empty_lines)

    def _is_all_caps(line: str) -> bool:
        letters = [c for c in line if c.isalpha()]
        if not letters:
            return False
        return all(c.isupper() for c in letters)

    def _is_title_case(line: str) -> bool:
        tokens = [tok for tok in re.split(r"\s+", line.strip()) if tok]
        if not tokens:
            return False
        alpha_tokens = [tok for tok in tokens if any(c.isalpha() for c in tok)]
        if not alpha_tokens:
            return False
        for tok in alpha_tokens:
            letters = [c for c in tok if c.isalpha()]
            if not letters:
                continue
            if not letters[0].isupper():
                return False
            if any(c.isupper() for c in letters[1:]):
                return False
        return True

    all_caps_count = sum(1 for line in non_empty_lines if _is_all_caps(line))
    title_case_count = sum(1 for line in non_empty_lines if _is_title_case(line))
    frac_heading_lines = (
        (all_caps_count + title_case_count) / non_empty_count
        if non_empty_count > 0
        else 0.0
    )

    first_10_lines = lines[:10]
    count_article_10 = sum(
        len(_ARTICLE_RE.findall(line))
        for line in first_10_lines
    )
    count_section_10 = sum(
        len(_SECTION_RE.findall(line))
        for line in first_10_lines
    )

    # Page number detection
    s = text.rsplit("\n", 1)[-1]
    m = _DIGIT_RE.match(s)
    if m:
        num = m.group(1)
        flag_is_all_digits = 1
        flag_is_less_than_order = int(int(num) < order)
    else:
        flag_is_all_digits = 0
        flag_is_less_than_order = 0

    # Legal document specific features
    count_section = text.lower().count("section")
    count_article = text.lower().count("article")
    num_all_caps = sum(1 for w in words if w.isalpha() and w.isupper())
    prop_word_cap = (
        sum(1 for w in words if w[:1].isupper()) / num_words if num_words > 0 else 0.0
    )

    # N-gram features
    bigrams = [" ".join(bg) for bg in zip(words, words[1:])]
    num_bigrams = len(bigrams)
    unique_bigrams = len(set(bigrams))
    prop_unique_bigrams = unique_bigrams / num_bigrams if num_bigrams > 0 else 0.0
    trigrams = [" ".join(tg) for tg in zip(words, words[1:], words[2:])]
    num_trigrams = len(trigrams)
    unique_trigrams = len(set(trigrams))
    prop_unique_trigrams = unique_trigrams / num_trigrams if num_trigrams > 0 else 0.0

    # HTML structure features
    num_tags = html.count("<")
    tag_to_text_ratio = num_tags / num_chars if num_chars > 0 else 0.0
    link_count = html.lower().count("<a ")
    img_count = html.lower().count("<img ")
    heading_tags = sum(html.lower().count(f"<h{i}") for i in range(1, 7))
    list_count = html.lower().count("<li")

    # Document structure features
    bullet_count = sum(1 for line in lines if line.strip().startswith("-"))

    # Legal boilerplate terms
    lower = text.lower()
    boilerplate_counts = [lower.count(term) for term in BOILERPLATE_TERMS]

    # Legal document keywords
    keyword_flags = [1.0 if kw in lower else 0.0 for kw in LEGAL_KEYWORDS]

    # Signature indicators
    sig_indicators = "by" in lower and "title" in lower

    # Punctuation analysis
    num_colon = text.count(":")
    num_period = text.count(".")
    num_consecutive_periods = max(
        (m.group().count(".") for m in _CONSEC_PERIOD_RE.finditer(text)), default=0
    )
    num_comma = text.count(",")
    total_punct = sum(c in string.punctuation for c in text)
    prop_colon = num_colon / total_punct if total_punct > 0 else 0.0
    prop_period = num_period / total_punct if total_punct > 0 else 0.0
    prop_comma = num_comma / total_punct if total_punct > 0 else 0.0

    # HTML structure analysis
    html_lower = html.lower()
    has_table = 1.0 if _HAS_TABLE_RE.search(html_lower) else 0.0
    count_p = html_lower.count("<p")
    count_div = html_lower.count("<div")

    # --- Extra cues for exhibits/back matter & signature blocks ---
    first_line = text.split("\n", 1)[0].strip()
    first_lower = first_line.lower()

    # Headings like "EXHIBIT A", "ANNEX I", "SCHEDULE 1.1(a)"
    exhibit_heading = 1.0 if _EXHIBIT_HEADING_RE.search(first_lower) else 0.0
    # Common back-matter/exhibit phrases
    has_counterparts = 1.0 if ("counterpart" in lower or "counterparts" in lower) else 0.0
    has_form_of = 1.0 if ("form of" in lower and "agreement" in lower) else 0.0
    signature_page_to = 1.0 if ("signature page to" in lower) else 0.0
    # Signature field cues
    sig_field_cues = sum(kw in lower for kw in ["name:", "title:", "its:", "date:"])
    # Long underscore runs (signature lines)
    underscore_runs = len(_UNDERSCORE_RUN_RE.findall(text))

    # Agreement-relative position features
    pages_remaining = max(safe_doc_page_count - 1.0 - safe_page_index, 0.0)
    rel_page_pos = safe_page_index / max(safe_doc_page_count - 1.0, 1.0)
    is_last_page = 1.0 if pages_remaining <= 0.0 else 0.0
    in_last_three_pages = 1.0 if pages_remaining <= 2.0 else 0.0
    in_last_decile = 1.0 if rel_page_pos >= 0.9 else 0.0
    in_last_quintile = 1.0 if rel_page_pos >= 0.8 else 0.0

    # Back-matter cues that often mimic agreement-level signature pages.
    has_executed_as_deed = 1.0 if "executed as a deed" in lower else 0.0
    has_signature_of_witness = 1.0 if "signature of witness" in lower else 0.0
    has_witness_address = 1.0 if "witness address" in lower else 0.0
    has_witness_occupation = 1.0 if "witness occupation" in lower else 0.0
    has_confidential_treatment = 1.0 if "confidential treatment" in lower else 0.0
    has_redaction_marker = 1.0 if _REDACTION_RE.search(text) else 0.0
    witness_block_combo = (
        1.0
        if has_executed_as_deed and (has_signature_of_witness or has_witness_occupation)
        else 0.0
    )

    # Compile feature vector
    features = [
        num_words,
        num_chars,
        avg_chars_per_word,
        prop_spaces,
        prop_digits,
        prop_newlines,
        prop_punct,
        num_lines,
        avg_line_len,
        frac_short_lines,
        frac_empty_lines,
        sig_line_count,
        frac_heading_lines,
        count_article_10,
        count_section_10,
        flag_is_all_digits,
        flag_is_less_than_order,
        count_section,
        count_article,
        num_all_caps,
        prop_word_cap,
        num_bigrams,
        unique_bigrams,
        prop_unique_bigrams,
        num_trigrams,
        unique_trigrams,
        prop_unique_trigrams,
        num_tags,
        tag_to_text_ratio,
        link_count,
        img_count,
        heading_tags,
        list_count,
        bullet_count,
        *boilerplate_counts,
        *keyword_flags,
        sig_indicators,
        prop_colon,
        prop_period,
        num_consecutive_periods,
        prop_comma,
        has_table,
        count_p,
        count_div,
        order,
        exhibit_heading,
        has_counterparts,
        has_form_of,
        signature_page_to,
        float(sig_field_cues),
        float(underscore_runs),
        safe_page_index,
        safe_doc_page_count,
        rel_page_pos,
        pages_remaining,
        is_last_page,
        in_last_three_pages,
        in_last_decile,
        in_last_quintile,
        has_executed_as_deed,
        has_signature_of_witness,
        has_witness_address,
        has_witness_occupation,
        has_confidential_treatment,
        has_redaction_marker,
        witness_block_combo,
    ]
    if len(features) != len(FEATURE_NAMES):
        raise ValueError("Feature vector length does not match FEATURE_NAMES.")
    return np.array(features, dtype=float)


def extract_base_feature_matrix(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required_cols = {"text", "html", "order", "agreement_uuid"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required feature columns: {sorted(missing)}")

    working = df.copy()
    working["text"] = working["text"].fillna("")
    working["html"] = working["html"].fillna("")
    working["order"] = working["order"].fillna(0.0).astype(float)
    working["agreement_uuid"] = working["agreement_uuid"].astype(str)

    sorted_df = working.sort_values(["agreement_uuid", "order"], kind="mergesort")
    sorted_df["_page_index"] = sorted_df.groupby("agreement_uuid").cumcount().astype(float)
    sorted_df["_doc_page_count"] = (
        sorted_df.groupby("agreement_uuid")["agreement_uuid"].transform("size").astype(float)
    )
    working = working.join(sorted_df[["_page_index", "_doc_page_count"]])
    if bool(working["_page_index"].isna().any()) or bool(working["_doc_page_count"].isna().any()):
        raise ValueError("Failed to compute page_index/doc_page_count for feature extraction.")

    features = np.vstack(
        cast(
            list[np.ndarray],
            list(
                Parallel(n_jobs=-1, prefer="threads")(
                    delayed(extract_features)(
                        text,
                        html,
                        order,
                        page_index,
                        doc_page_count,
                    )
                    for text, html, order, page_index, doc_page_count in zip(
                        working["text"],
                        working["html"],
                        working["order"],
                        working["_page_index"],
                        working["_doc_page_count"],
                    )
                )
            ),
        )
    ).astype(np.float32, copy=False)

    agreements = working["agreement_uuid"].to_numpy(dtype=str)
    orders = working["order"].to_numpy(dtype=np.float32)
    return features, agreements, orders


def augment_with_neighbor_context(
    base_features: np.ndarray,
    agreement_uuids: np.ndarray,
    orders: np.ndarray,
    *,
    base_feature_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    names = list(base_feature_names) if base_feature_names is not None else feature_name_list()
    if base_features.shape[1] != len(names):
        raise ValueError("base_features width must match base feature names.")

    name_to_idx = {name: idx for idx, name in enumerate(names)}
    missing = [f for f in NEIGHBOR_CONTEXT_SOURCE_FEATURES if f not in name_to_idx]
    if missing:
        raise ValueError(f"Missing source features for neighbor context: {missing}")

    n_rows = base_features.shape[0]
    sort_df = pd.DataFrame(
        {
            "agreement_uuid": agreement_uuids.astype(str),
            "order": orders.astype(float),
            "row_idx": np.arange(n_rows, dtype=np.int64),
        }
    ).sort_values(["agreement_uuid", "order"], kind="mergesort")

    sorted_idx = sort_df["row_idx"].to_numpy(dtype=np.int64)
    sorted_agreements = sort_df["agreement_uuid"].to_numpy(dtype=str)

    same_prev = np.zeros(n_rows, dtype=bool)
    same_prev[1:] = sorted_agreements[1:] == sorted_agreements[:-1]
    same_next = np.zeros(n_rows, dtype=bool)
    same_next[:-1] = sorted_agreements[:-1] == sorted_agreements[1:]

    add_sorted_parts: list[np.ndarray] = []
    for feature in NEIGHBOR_CONTEXT_SOURCE_FEATURES:
        col_sorted = base_features[sorted_idx, name_to_idx[feature]]
        prev_sorted = np.zeros(n_rows, dtype=np.float32)
        next_sorted = np.zeros(n_rows, dtype=np.float32)
        prev_sorted[1:] = np.where(same_prev[1:], col_sorted[:-1], 0.0)
        next_sorted[:-1] = np.where(same_next[:-1], col_sorted[1:], 0.0)
        add_sorted_parts.append(prev_sorted)
        add_sorted_parts.append(next_sorted)

    num_words_sorted = base_features[sorted_idx, name_to_idx["num_words"]]
    rel_pos_sorted = base_features[sorted_idx, name_to_idx["rel_page_pos"]]
    delta_prev_words = np.zeros(n_rows, dtype=np.float32)
    delta_next_words = np.zeros(n_rows, dtype=np.float32)
    delta_prev_rel = np.zeros(n_rows, dtype=np.float32)
    delta_next_rel = np.zeros(n_rows, dtype=np.float32)
    delta_prev_words[1:] = np.where(
        same_prev[1:], np.abs(num_words_sorted[1:] - num_words_sorted[:-1]), 0.0
    )
    delta_next_words[:-1] = np.where(
        same_next[:-1], np.abs(num_words_sorted[:-1] - num_words_sorted[1:]), 0.0
    )
    delta_prev_rel[1:] = np.where(
        same_prev[1:], np.abs(rel_pos_sorted[1:] - rel_pos_sorted[:-1]), 0.0
    )
    delta_next_rel[:-1] = np.where(
        same_next[:-1], np.abs(rel_pos_sorted[:-1] - rel_pos_sorted[1:]), 0.0
    )
    add_sorted_parts.extend(
        [delta_prev_words, delta_next_words, delta_prev_rel, delta_next_rel]
    )

    add_sorted = np.column_stack(add_sorted_parts).astype(np.float32, copy=False)
    add_unsorted = np.zeros((n_rows, add_sorted.shape[1]), dtype=np.float32)
    add_unsorted[sorted_idx] = add_sorted
    full_features = np.hstack([base_features, add_unsorted]).astype(np.float32, copy=False)
    full_names = names + neighbor_context_feature_name_list()
    if full_features.shape[1] != len(full_names):
        raise ValueError("Neighbor-augmented feature matrix width mismatch.")
    return full_features, full_names


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    include_neighbor_context: bool,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    base_features, agreements, orders = extract_base_feature_matrix(df)
    base_names = feature_name_list()
    if not include_neighbor_context:
        return base_features, base_names, agreements
    full_features, full_names = augment_with_neighbor_context(
        base_features, agreements, orders, base_feature_names=base_names
    )
    return full_features, full_names, agreements


def hard_negative_back_matter_mask(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: Sequence[str],
    *,
    back_label_idx: int,
) -> np.ndarray:
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must have the same row count.")
    if features.shape[1] != len(feature_names):
        raise ValueError("features width must match feature_names length.")

    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    required = [
        "kw_slash_s",
        "sig_field_cues",
        "has_executed_as_deed",
        "has_signature_of_witness",
        "signature_page_to",
    ]
    missing = [name for name in required if name not in name_to_idx]
    if missing:
        raise ValueError(f"Missing features required for hard-negative mask: {missing}")

    back_rows = labels == back_label_idx
    sig_like = (
        (features[:, name_to_idx["kw_slash_s"]] > 0.0)
        | (features[:, name_to_idx["sig_field_cues"]] >= 2.0)
        | (features[:, name_to_idx["has_executed_as_deed"]] > 0.0)
        | (features[:, name_to_idx["has_signature_of_witness"]] > 0.0)
        | (features[:, name_to_idx["signature_page_to"]] > 0.0)
    )
    return back_rows & sig_like
