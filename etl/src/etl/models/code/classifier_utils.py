"""
Feature extraction utilities for page classification.

This module provides functions to extract features from text and HTML content
for use in page classification models.
"""

import re
import string
from typing import List

import numpy as np


def extract_features(text: str, html: str, order: float) -> np.ndarray:
    """
    Extract features from text and HTML content for page classification.
    
    Args:
        text: Raw text content of the page
        html: HTML content of the page
        order: Page order/sequence number
        
    Returns:
        numpy.ndarray: Feature vector for classification
    """
    # Handle NaN values
    text = str(text) if text == text else ""
    html = str(html) if html == html else ""

    # Basic text statistics
    num_chars = len(text)
    words = text.split()
    num_words = len(words)
    avg_chars_per_word = num_chars / num_words if num_words > 0 else 0.0
    prop_spaces = text.count(" ") / num_chars if num_chars > 0 else 0.0
    prop_digits = sum(c.isdigit() for c in text) / num_chars if num_chars > 0 else 0.0
    prop_newlines = text.count("\\n") / num_chars if num_chars > 0 else 0.0
    prop_punct = (
        sum(c in string.punctuation for c in text) / num_chars if num_chars > 0 else 0.0
    )

    # Page number detection
    _DIGIT_RE = re.compile(r"^[\-\s—]*(\d+)[\-\s—]*$")
    s = text.rsplit("\\n", 1)[-1]
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
    bullet_count = sum(1 for line in text.split("\\n") if line.strip().startswith("-"))

    # Legal boilerplate terms
    boilerplate_terms = ["hereto", "herein", "hereby", "thereof", "wherein"]
    boilerplate_counts = [text.lower().count(t) for t in boilerplate_terms]

    # Legal document keywords
    legal_keywords = [
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
        "/s/",
        "intentionally blank",
        "page follows]",
        "page follows.]",
        "by:",
    ]
    keyword_flags = [1.0 if kw in text.lower() else 0.0 for kw in legal_keywords]

    # Signature indicators
    sig_indicators = "by" in text.lower() and "title" in text.lower()

    # Punctuation analysis
    num_colon = text.count(":")
    num_period = text.count(".")
    num_consecutive_periods = max(
        (m.group().count(".") for m in re.finditer(r"(?: *\.)+", text)), default=0
    )
    num_comma = text.count(",")
    total_punct = sum(c in string.punctuation for c in text)
    prop_colon = num_colon / total_punct if total_punct > 0 else 0.0
    prop_period = num_period / total_punct if total_punct > 0 else 0.0
    prop_comma = num_comma / total_punct if total_punct > 0 else 0.0

    # HTML structure analysis
    has_table = 1.0 if re.search(r"</?(table|tr|td)", html.lower()) else 0.0
    count_p = html.lower().count("<p")
    count_div = html.lower().count("<div")

    # Compile feature vector
    features = [
        num_words,
        num_chars,
        avg_chars_per_word,
        prop_spaces,
        prop_digits,
        prop_newlines,
        prop_punct,
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
    ]
    return np.array(features, dtype=float)
